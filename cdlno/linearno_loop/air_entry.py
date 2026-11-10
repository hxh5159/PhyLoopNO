"""AirfRANS loop adapter; native sampling, MSE, metrics and ensemble order."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from cdlno.linearno import air_entry as pure
from cdlno.linearno.air_entry import _data_root, _data_spec, _load_dataset, _restore_coef_norm, _pressure_rL2_dataset
from cdlno.linearno.checkpoint import resume_state, restore_random_state, strict_load, sha256
from cdlno.linearno.schema import unpack_state
from cdlno.linearno.profiles import digest
from cdlno.training_state import _optimizer_signature, _atomic
from linearno_loop.versioning import make_metadata, read_metadata, write_metadata
from linearno_loop.contracts import require_equal
from .industrial_state import construct, generators, data_contract, restore_training, inspect_members, export_state, member_seed
from .versioning import checkpoint_api, provenance

class AirRun(pure.AirRun):

    def __init__(self, args, config, data_dir, coef_norm, recorder, *, manifest=None, evaluation=False):
        self.args, self.config, self.data_dir, self.coef_norm = (args, config, Path(data_dir), coef_norm)
        self.directory = Path(args.linearno_run_dir)
        self.recorder, self.evaluation = (recorder, evaluation)
        self.manifest = manifest if manifest is not None else self._read_manifest()
        self.data = copy.deepcopy(_data_spec(self.data_dir, self.manifest, config, args.task))
        checksum = digest(self.data['checksums'])
        self.data['checksums']['normalizer_fit_dataset'] = checksum
        self.normalizers = pure._normalizer_spec(coef_norm, checksum)
        self.checkpoint = checkpoint_api(args._linearno_loop_config)
        self.current_provenance = provenance(args._linearno_loop_config, task='airfrans')
        self.provenance = self.current_provenance
        self.generators = generators(args)
        self.current_member = 0
        self.completed_epoch = 0
        self.steps_per_epoch = self.data['steps_per_epoch']
        self.data['ensemble_initialization'] = [dict(member=i, seed=member_seed(args, i)) for i in range(args.nmodel)]
        if hasattr(args, '_linearno_metadata'):
            saved = args._linearno_metadata
            if args.resume and saved['provenance_spec']['source_sha256'] != self.current_provenance['source_sha256']:
                raise ValueError('resume source differs')
            self.provenance = saved['provenance_spec']
            native = copy.deepcopy(saved['data_spec']['runtime'])
            native.pop('optimizer_signature')
            expected = data_contract(args._linearno_loop_config, self.data)
            require_equal(expected, {**saved['data_spec'], 'runtime': native}, 'saved.data_spec')
            require_equal(self.normalizers, saved['normalizer_spec'], 'saved.normalizer_spec')
            self.data['optimizer_signature'] = saved['data_spec']['runtime']['optimizer_signature']

    @property
    def model_spec(self):
        return self.args._linearno_model_spec

    def loader_kwargs(self, split):
        return dict(generator=self.generators[split])

    def _resume_state(self, optimizer, scheduler, epoch, sampler):
        return self.checkpoint.resume_state(optimizer, scheduler, epoch, self.steps_per_epoch, self.args.nb_epochs, self.generators, sampler)

    def _metadata(self, state, model=None):
        if model is None:
            raise ValueError('metadata requires a constructed model')
        return make_metadata(self.args._linearno_loop_config, data_spec=data_contract(self.args._linearno_loop_config, self.data), normalizer_spec=self.normalizers, provenance_spec=self.provenance, resume_state=state, ensemble_manifest=[], parameter_measurement=self.checkpoint.measure_parameters(model, self.args._linearno_loop_config))

    def prepare(self, model, optimizer, scheduler, train_dataset, val_dataset, criterion, reg, val_iter, val_sample):
        objective = self.config['values']['objective']
        if criterion != 'MSE_weighted' or objective['kind'] != 'MSE' or objective['space'] != 'normalized' or (reg != objective['surface_weight']):
            raise ValueError('Air native objective differs from profile')
        if len(train_dataset) != self.steps_per_epoch or scheduler.total_steps != self.data['scheduler_total_steps']:
            raise ValueError('Air loader/scheduler differs from metadata')
        self.bind(optimizer, scheduler)
        signature = json.loads(json.dumps(_optimizer_signature(model, optimizer)))
        if 'optimizer_signature' in self.data:
            require_equal(self.data['optimizer_signature'], signature, 'optimizer_signature')
        self.data['optimizer_signature'] = signature
        initial = self._metadata(self._resume_state(optimizer, scheduler, 0, dict(member=self.current_member, epoch_boundary=True)), model)
        root = self.directory / 'architecture.json'
        if not root.exists():
            write_metadata(root, initial)
        else:
            stored = read_metadata(root)
            for key in ('data_spec', 'normalizer_spec', 'resolved_config'):
                require_equal(stored[key], initial[key], key)
        self.member_dir.mkdir(exist_ok=True)
        sidecar = self.member_dir / 'architecture.json'
        if not sidecar.exists():
            write_metadata(sidecar, initial)
        self.completed_epoch = 0
        if not self.args.resume or not (self.member_dir / 'checkpoints/latest.json').exists():
            return (0, None)
        saved, path = self.checkpoint.inspect_checkpoint(self.member_dir, self.args.checkpoint, expected=self.args._linearno_loop_config)
        if unpack_state(saved['resume_state']['sampler_state'])['member'] != self.current_member:
            raise ValueError('wrong member')
        self.completed_epoch = restore_training(saved, path, model, optimizer, scheduler, steps=self.steps_per_epoch, generators=self.generators, current_provenance=self.current_provenance)
        history = unpack_state(saved['resume_state']['sampler_state']).get('history')
        if history is None or any((len(c) != self.completed_epoch for c in history['curves'][:4])):
            raise ValueError('checkpoint epoch history mismatch')
        return (self.completed_epoch, history)

    def complete_epoch(self, epoch, model, history):
        if epoch != self.completed_epoch + 1 or self.scheduler.last_epoch != epoch * self.steps_per_epoch:
            raise ValueError('Air epoch/scheduler cadence mismatch')
        state = self._resume_state(self.optimizer, self.scheduler, epoch, dict(member=self.current_member, epoch_boundary=True, history=pure._json_history(history)))
        self.checkpoint.save_pair(self.member_dir, model, self._metadata(state, model))
        self.completed_epoch = epoch
        _atomic(self.member_dir / 'history.json', pure._json_history(history), json_file=True)

    def export_final(self, model):
        export_state(model, self.member_dir / 'model')

    def load_models(self):
        root = read_metadata(self.directory / 'architecture.json')
        pairs = inspect_members(self.directory, root, evaluation=True, selector=self.args.checkpoint, checkpoint_module=self.checkpoint)
        models = []
        for index, (metadata, path) in enumerate(pairs):
            model = construct(self.args, index)
            _, weights = self.checkpoint.read_pair(path, expected=self.args._linearno_loop_config)
            strict_load(model, weights)
            models.append(model.to(self.args.device).eval())
        return models

    def finish_ensemble(self, models):
        if len(models) != self.args.nmodel or len({id(m.loop) for m in models}) != len(models):
            raise ValueError('ensemble must contain independent cores')
        ids = [{id(p) for p in m.parameters()} for m in models]
        if any((ids[i] & ids[j] for i in range(len(ids)) for j in range(i))):
            raise ValueError('ensemble shares parameters')
        rows = []
        for i, model in enumerate(models):
            member = self.directory / f'member_{i:03d}'
            _, path = self.checkpoint.inspect_checkpoint(member, 'final', expected=self.args._linearno_loop_config)
            _, weights = self.checkpoint.read_pair(path)
            strict_load(model, weights)
            manifest = json.loads(path.read_text())
            weight = member / manifest['weights']['path']
            rows.append(dict(member_id=f'member_{i:03d}', order=i, path=str(weight.relative_to(self.directory)), sha256=sha256(weight), format='state_dict'))
        value = dict(family='linearno_loop', config_hash=self.args._linearno_loop_config['config_hash'], checkpoint_role='final', members=rows)
        path = self.directory / 'ensemble.json'
        if path.exists():
            require_equal(json.loads(path.read_text()), value, 'ensemble')
        else:
            _atomic(path, value, json_file=True)
        _atomic(self.directory / 'ensemble_state_dict.pth', [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in models])

def run_cli(args):
    from cdlno.experiment import start, finish
    import train as air_train
    import utils.metrics as metrics
    config = args._linearno_config
    hparams = pure._hparams(config)
    recorder = start(args, 'airfrans', evaluation=bool(args.eval), hparams=hparams)
    data_dir = _data_root(args.my_path, evaluation=bool(args.eval))
    saved_coef = _restore_coef_norm(getattr(args, '_linearno_metadata', None))
    if args.eval:
        manifest, coef = _load_dataset(data_dir, args, train=False, coef_norm=saved_coef)
    else:
        manifest, train_dataset, val_dataset, coef = _load_dataset(data_dir, args, train=True, coef_norm=saved_coef)
        hparams['total_steps'] = (len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs']
    args._linearno_normalizers = {'coef_norm': coef}
    run = AirRun(args, config, data_dir, coef, recorder, manifest=manifest, evaluation=bool(args.eval))
    device = torch.device(args.device)
    if args.eval:
        models = run.load_models()
        result_dir = run.eval_dir()
        scores = metrics.Results_test(str(device), [models], [hparams], coef, str(data_dir), str(result_dir), n_test=3, criterion='MSE', s=args.task + '_test' if args.task != 'scarce' else 'full_test')
        pressure = _pressure_rL2_dataset(str(device), models, hparams, coef, data_dir, manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test'])
        (result_dir / 'pressure_rL2.json').write_text(json.dumps(pressure, indent=2) + '\n')
        np.save(result_dir / 'true_coefs', scores[0])
        np.save(result_dir / 'pred_coefs_mean', scores[1])
        np.save(result_dir / 'pred_coefs_std', scores[2])
        recorder.record_metrics(dict(result_dir=str(result_dir), field_metric='normalized four-channel MSE', pressure_field_metric='physical pressure-channel rL2 by volume/surface region', pressure_rL2=pressure, force_metric='relative coefficient error and Spearman', checkpoint=args.checkpoint))
        finish(args)
        return result_dir
    models = []
    for index in range(args.nmodel):
        run.current_member = index
        model = construct(args, index).to(device)
        if args.resume and (run.member_dir / 'checkpoints/latest.json').exists():
            saved, path = run.checkpoint.inspect_checkpoint(run.member_dir, 'latest', expected=args._linearno_loop_config)
            if saved['provenance_spec']['source_sha256'] != run.current_provenance['source_sha256']:
                raise ValueError('resume source differs')
            if saved['resume_state']['checkpoint_role'] == 'final':
                _, weights = run.checkpoint.read_pair(path)
                strict_load(model, weights)
                run.export_final(model)
                restore_random_state(saved['resume_state'], run.generators)
                models.append(model)
                continue
        recorder.attach_model(model, hparams=hparams, member=index, protocol=dict(family='linearno_loop', profile=config['profile'], objective=config['values']['objective'], steps_per_epoch=len(train_dataset) // hparams['batch_size'] + 1, total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs']))
        from cdlno.linearno.air_visualization import AirFields
        if not hasattr(recorder, '_field_visualizers'):
            recorder._field_visualizers = {}
        recorder._field_visualizers[index] = AirFields(run.directory, 'airfrans', 'Looped LinearNO', seed=args.seed, member=index)
        model = air_train.main(str(device), train_dataset, val_dataset, model, hparams, str(run.member_dir), criterion='MSE_weighted', reg=args.weight, val_iter=10, name_mod='LinearNO', val_sample=True, record=recorder, record_member=index, visualization_norm=coef, linearno_run=run)
        saved, _ = run.checkpoint.inspect_checkpoint(run.member_dir, 'final', expected=args._linearno_loop_config)
        restore_random_state(saved['resume_state'], run.generators)
        models.append(model)
    run.finish_ensemble(models)
    finish(args)
    return run.directory
