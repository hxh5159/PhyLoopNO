"""AirfRANS LinearNO entry and native state-dict checkpoint adapter.

The AirfRANS project keeps its original loader, sampling loop, optimizer and
metric implementation.  This module only resolves the LinearNO profile,
reconstructs the task model from metadata, and stores strict state-dict epoch
pairs.  It is imported only after ``--model LinearNO`` is selected.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from uuid import uuid4

import numpy as np
import torch

from .airfrans import AirfRANSLinearNO
from .checkpoint import (inspect_checkpoint, read_pair, restore_random_state, resume_state,
                         save_pair, strict_load)
from .profiles import PROFILES, DEFAULT_PROFILE, digest, resolve_config, require_resolved_objective
from .schema import (make_metadata, normalizer_record, numerical_state,
                     read_metadata, restore_numerical_state, write_metadata,
                     unpack_state)

ROOT = Path(__file__).resolve().parents[2]


def _has(tokens, *names):
    return any(token == name or token.startswith(name + '=') for token in tokens for name in names)


def _value(args, name):
    value = getattr(args, name, None)
    return value


def _data_root(value, *, evaluation=False):
    path = Path(value).expanduser().resolve()
    if (path / 'manifest.json').is_file():
        return path
    nested = path / 'Dataset'
    if (nested / 'manifest.json').is_file():
        return nested
    action = 'evaluation' if evaluation else 'training'
    raise ValueError(f'AirfRANS {action} data root must contain manifest.json or Dataset/manifest.json: {path}')


def _constructor_kwargs(config):
    model = config['values']['model']
    return dict(space_dim=model['space_dim'], n_layers=model['layers'],
                n_hidden=model['hidden'], dropout=model['dropout'],
                n_head=model['heads'], act=model['activation'],
                mlp_ratio=model['ffn_ratio'], fun_dim=model['fun_dim'],
                out_dim=model['out_dim'], linearno_rank=model['linearno_rank'],
                ref=model['ref'], unified_pos=model['unified_pos'], linear=True)


def _profile_explicit(args, tokens):
    """Map only values actually present on argv; old parser defaults are ignored."""
    values = {}
    fields = (
        (('--linearno-variant',), 'model.linearno_variant', 'linearno_variant'),
        (('--linearno-rank',), 'model.linearno_rank', 'linearno_rank'),
        (('--linearno-hidden',), 'model.hidden', 'linearno_hidden'),
        (('--linearno-layers',), 'model.layers', 'linearno_layers'),
        (('--linearno-heads',), 'model.heads', 'linearno_heads'),
        (('--linearno-ffn-ratio',), 'model.ffn_ratio', 'linearno_ffn_ratio'),
        (('--linearno-dropout',), 'model.dropout', 'linearno_dropout'),
        (('--linearno-ref',), 'model.ref', 'linearno_ref'),
        (('--linearno-unified-pos',), 'model.unified_pos', 'linearno_unified_pos'),
        (('--nb_epochs', '--epochs'), 'training.epochs', 'nb_epochs'),
        (('--batch_size', '--batch-size'), 'training.batch_size', 'batch_size'),
        (('--lr',), 'training.lr', 'lr'),
        (('--nmodel', '-n'), 'training.nmodel', 'nmodel'),
        (('--weight', '-w'), 'objective.surface_weight', 'weight'),
        (('--subsampling',), 'training.subsampling', 'subsampling'),
        (('--r',), 'training.r', 'r'),
        (('--max_neighbors', '--max-neighbors'), 'training.max_neighbors', 'max_neighbors'),
        (('--debug',), 'training.debug', 'debug'),
        (('--seed',), 'runtime.seed', 'seed'),
        (('--save-name', '--save_name'), 'runtime.save_name', 'save_name'),
    )
    for flags, path, name in fields:
        if _has(tokens, *flags):
            value = getattr(args, name, None)
            if path == 'model.unified_pos' and value is not None:
                value = bool(value)
            values[path] = value
    return values


def _set_resolved_args(args, config):
    model = config['values']['model']
    training = config['values']['training']
    args.linearno_profile = config['profile']
    args.linearno_variant = model['linearno_variant']
    args.linearno_rank = model['linearno_rank']
    args.n_hidden = model['hidden']
    args.n_layers = model['layers']
    args.n_heads = model['heads']
    args.mlp_ratio = model['ffn_ratio']
    args.dropout = model['dropout']
    args.unified_pos = int(model['unified_pos'])
    args.ref = model['ref']
    args.nb_epochs = training['epochs']
    args.batch_size = training['batch_size']
    args.lr = training['lr']
    args.nmodel = training['nmodel']
    args.weight = config['values']['objective']['surface_weight']
    args.subsampling = training['subsampling']
    args.r = training['r']
    args.max_neighbors = training['max_neighbors']
    args.debug = training['debug']
    args.save_name = config['values']['runtime']['save_name']
    args.seed = config['values']['runtime']['seed']
    args.linearno_family = 'linearno'
    args.linearno_task = 'airfrans'
    args._linearno_config = config
    args._linearno_model_spec = dict(class_path='cdlno.linearno.airfrans.AirfRANSLinearNO',
                                     constructor_kwargs=_constructor_kwargs(config))


def check_structure(metadata):
    expected = dict(class_path='cdlno.linearno.airfrans.AirfRANSLinearNO',
                    constructor_kwargs=_constructor_kwargs(metadata['profile_spec']))
    if metadata['profile_spec']['task'] != 'airfrans' or metadata['model_spec'] != expected:
        raise ValueError('AirfRANS model_spec differs from resolved profile structure')
    if metadata['objective_spec'] != metadata['profile_spec']['values']['objective']:
        raise ValueError('AirfRANS objective differs from resolved profile')


def parse_args(parser, tokens, *, evaluation=False):
    parser.add_argument('--linearno-profile', choices=PROFILES, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-variant', choices=('airfrans',), default=argparse.SUPPRESS)
    parser.add_argument('--linearno-rank', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-hidden', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-layers', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-heads', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-ffn-ratio', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-dropout', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-ref', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-unified-pos', type=int, choices=(0, 1), default=argparse.SUPPRESS)
    parser.add_argument('--subsampling', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--r', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--max-neighbors', '--max_neighbors', dest='max_neighbors', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--debug', type=int, choices=(0,), default=argparse.SUPPRESS)
    # These aliases belong only to the LinearNO branch.  The historical AirfRANS
    # parser keeps its original --nb_epochs/--batch_size/--score interface.
    parser.add_argument('--epochs', dest='nb_epochs', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--save-name', '--save_name', dest='save_name', default=argparse.SUPPRESS)
    parser.add_argument('--experiment-dir', '--linearno-run-dir', dest='linearno_run_dir',
                        type=Path, default=argparse.SUPPRESS)
    parser.add_argument('--eval', type=int, choices=(0, 1), default=argparse.SUPPRESS)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--checkpoint', default='final')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args(tokens)
    # main_evaluation.py has no --eval flag; its evaluation=True call is the
    # authoritative action selector.  The training entry defaults to train.
    args.eval = 1 if evaluation else int(getattr(args, 'eval', 0))
    # Resume is a continuation action; when the user did not explicitly pick
    # a checkpoint, consume the committed latest pointer instead of looking for
    # the final archive first.  Evaluation keeps its final-checkpoint default.
    if args.resume and not _has(tokens, '--checkpoint'):
        args.checkpoint = 'latest'
    if not hasattr(args, 'save_name'):
        args.save_name = None
    explicit = _profile_explicit(args, tokens)
    run_dir = getattr(args, 'linearno_run_dir', None) or getattr(args, 'run_dir', None)
    profile = getattr(args, 'linearno_profile', DEFAULT_PROFILE)
    if getattr(args, 'task', 'full') not in ('full', 'scarce', 'reynolds', 'aoa'):
        parser.error('LinearNO AirfRANS task must be full, scarce, reynolds or aoa')
    if args.eval and args.resume:
        parser.error('evaluation and resume are separate actions')
    if args.eval or args.resume:
        if run_dir is None:
            parser.error('LinearNO evaluation/resume requires --experiment-dir')
        metadata_path = Path(run_dir).resolve() / 'architecture.json'
        try:
            metadata = read_metadata(metadata_path, constructor=AirfRANSLinearNO)
            check_structure(metadata)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        saved = metadata['profile_spec']
        if 'linearno_profile' in vars(args) and profile != saved['profile']:
            parser.error('explicit --linearno-profile conflicts with saved metadata')
        for path, value in explicit.items():
            section, field = path.split('.')
            if saved['values'][section][field] != value:
                parser.error(f'explicit {path} conflicts with saved metadata')
        config = saved
        args._linearno_metadata = metadata
        saved_task = metadata['data_spec']['task_variant']
        if _has(tokens, '--task', '-t') and args.task != saved_task:
            parser.error('explicit task conflicts with saved data split')
        args.task = saved_task
        try:
            _, checkpoint_path = inspect_checkpoint(Path(run_dir) / 'member_000',
                                                     args.checkpoint, AirfRANSLinearNO)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        args._linearno_checkpoint = checkpoint_path
    else:
        try:
            config = resolve_config('airfrans', profile, explicit=explicit)
        except ValueError as error:
            parser.error(str(error))
    try:
        require_resolved_objective(config)
        if config['values']['training']['batch_size'] != 1:
            raise ValueError('AirfRANS LinearNO supports batch_size=1 only')
        if args.resume and config['values']['training']['nmodel'] > 1 and args.checkpoint != 'latest':
            raise ValueError('ensemble resume requires --checkpoint latest')
    except ValueError as error:
        parser.error(str(error))
    _set_resolved_args(args, config)
    args.linearno_run_dir = Path(run_dir).resolve() if run_dir is not None else None
    if args.linearno_run_dir is None:
        root = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT / 'output'))
        leaf = args.save_name or f'seed{args.seed}_{uuid4().hex[:10]}'
        args.linearno_run_dir = (root / 'airfrans' / 'linearno' / config['profile'] /
                                 f"{config['values']['model']['linearno_variant']}_M{config['values']['model']['linearno_rank']}_eval{digest(config['values']['evaluation'])[:10]}" /
                                 leaf).resolve()
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    if args.gpu is not None:
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.data_path = args.my_path
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _provenance():
    from cdlno.provenance import provenance
    return provenance(task='airfrans')


def _hparams(config):
    training = config['values']['training']
    # LinearNO does not construct radius graphs.  Omitting these two keys also
    # makes the original Infer_test choose its no-edge branch via KeyError.
    return dict(batch_size=training['batch_size'], nb_epochs=training['epochs'],
                lr=training['lr'], subsampling=training['subsampling'])


def _normalizer_spec(coef_norm, manifest_hash):
    if coef_norm is None or len(coef_norm) != 4:
        raise ValueError('AirfRANS requires the train-fit four-part coef_norm')
    names = dict(input_mean=coef_norm[0], input_std=coef_norm[1],
                 output_mean=coef_norm[2], output_std=coef_norm[3])
    return dict(policy='saved_train_fit', records={'coef_norm': normalizer_record(
        names, fit_split='train manifest excluding final 10% validation',
        data_checksum=manifest_hash, algorithm='AirfRANS Dataset UnitTransformer mean/std + 1e-8')})


def _data_spec(data_dir, manifest, config, task_name='full'):
    if not isinstance(manifest, dict):
        raise ValueError('AirfRANS manifest must be loaded before metadata construction')
    manifest_hash = _sha256(Path(data_dir) / 'manifest.json')
    task = config['task']
    names = manifest.get(task_name + '_train', [])
    validation_count = int(.1 * len(names))
    train_count = len(names) - validation_count
    batch_size = config['values']['training']['batch_size']
    actual_steps = (train_count + batch_size - 1) // batch_size
    release_steps = train_count // batch_size + 1
    checksums = {'manifest.json': manifest_hash}
    test_key = 'full_test' if task_name == 'scarce' else task_name + '_test'
    for name in dict.fromkeys(names + manifest[test_key]):
        for suffix in ('_internal.vtu', '_aerofoil.vtp'):
            relative = Path(name) / (name + suffix)
            if Path(name).name != name or not (Path(data_dir) / relative).is_file():
                raise ValueError(f'AirfRANS required dataset file missing/invalid: {relative}')
            checksums[str(relative)] = _sha256(Path(data_dir) / relative)
    return dict(split='full_train last10% validation; full_test evaluation; task split selected by --task',
                checksums=checksums,
                sampling='train random subsampling 32000; eval random subsets until all points covered; repeated-point mean',
                manifest_hash=manifest_hash, task=task, task_variant=task_name,
                manifest_keys=sorted(manifest), train_count=train_count,
                validation_count=validation_count, steps_per_epoch=actual_steps,
                scheduler_total_steps=release_steps * config['values']['training']['epochs'],
                scheduler_formula='(len(train_dataset)//batch_size+1)*epochs')


class AirRun:
    """One AirfRANS run; ``member`` state lives in its own strict pair tree."""
    def __init__(self, args, config, data_dir, coef_norm, recorder, *, manifest=None, evaluation=False):
        self.args, self.config, self.data_dir, self.coef_norm = args, config, Path(data_dir), coef_norm
        self.recorder, self.evaluation = recorder, evaluation
        self.directory = Path(args.linearno_run_dir).resolve()
        self.manifest = manifest if manifest is not None else self._read_manifest()
        self.data = copy.deepcopy(_data_spec(self.data_dir, self.manifest, config, args.task))
        checksum = digest(self.data['checksums'])
        self.data['checksums']['normalizer_fit_dataset'] = checksum
        self.normalizers = _normalizer_spec(coef_norm, checksum)
        self.provenance = _provenance()
        self.current_member = 0
        self._member_metadata = None
        self.completed_epoch = 0
        self.steps_per_epoch = None

        if (evaluation or getattr(args, 'resume', False)) and hasattr(args, '_linearno_metadata'):
            saved = args._linearno_metadata
            if saved['data_spec'] != self.data:
                raise ValueError('AirfRANS data manifest/split differs from saved metadata')
            if saved['normalizer_spec'] != self.normalizers:
                raise ValueError('AirfRANS normalizer differs from saved train-fit metadata')

    def _read_manifest(self):
        with (self.data_dir / 'manifest.json').open() as stream:
            return json.load(stream)

    @property
    def model_spec(self):
        return dict(class_path='cdlno.linearno.airfrans.AirfRANSLinearNO',
                    constructor_kwargs=_constructor_kwargs(self.config))

    @property
    def member_dir(self):
        return self.directory / f'member_{self.current_member:03d}'

    def _resume_zero(self, optimizer, scheduler):
        return resume_state(optimizer, scheduler, 0,
                            self.steps_per_epoch, self.config['values']['training']['epochs'], {}, {})

    def _metadata(self, resume):
        return make_metadata(profile_spec=self.config, constructor=AirfRANSLinearNO,
            model_spec=self.model_spec,
            data_spec=self.data, objective_spec=self.config['values']['objective'],
            evaluation_spec=self.config['values']['evaluation'], provenance_spec=self.provenance,
            normalizer_spec=self.normalizers, resume_state=resume, ensemble_manifest=[])

    def _write_root_metadata(self, optimizer, scheduler):
        path = self.directory / 'architecture.json'
        if path.exists():
            saved = read_metadata(path, constructor=AirfRANSLinearNO)
            if saved['model_spec'] != self.model_spec or saved['profile_spec'] != self.config:
                raise ValueError('AirfRANS run metadata structure/profile mismatch')
            if saved['data_spec'] != self.data or saved['normalizer_spec'] != self.normalizers:
                raise ValueError('AirfRANS run metadata data/normalizer mismatch')
            return saved
        metadata = self._metadata(self._resume_zero(optimizer, scheduler))
        write_metadata(path, metadata, constructor=AirfRANSLinearNO)
        return metadata

    def prepare(self, model, optimizer, scheduler, train_dataset, val_dataset, criterion, reg, val_iter, val_sample):
        expected = self.config['values']['objective']
        if criterion != 'MSE_weighted' or expected['kind'] != 'MSE' or expected['space'] != 'normalized' or reg != expected['surface_weight']:
            raise ValueError('AirfRANS actual MSE criterion/weight differs from metadata')
        if len(train_dataset) < 1:
            raise ValueError('AirfRANS LinearNO requires at least one training sample')
        # The optimizer/scheduler are owned by the native train loop.  Binding
        # them here makes each checkpoint self-contained and reentrant.
        self.bind(optimizer, scheduler)
        actual_steps = (len(train_dataset) + self.config['values']['training']['batch_size'] - 1) // self.config['values']['training']['batch_size']
        if actual_steps != self.data['steps_per_epoch']:
            raise ValueError('AirfRANS loader length differs from metadata train split')
        self.steps_per_epoch = actual_steps
        self.completed_epoch = 0
        self._write_root_metadata(optimizer, scheduler)
        self.member_dir.mkdir(parents=True, exist_ok=True)
        self._member_metadata = self._metadata(self._resume_zero(optimizer, scheduler))
        if not self.args.resume:
            return 0, None
        pointer = self.member_dir / 'checkpoints' / 'latest.json'
        if not pointer.is_file() and self.current_member > 0:
            # A sequential ensemble may be interrupted before this member starts.
            # The preceding member's committed final RNG is restored by run_cli.
            inspect_checkpoint(self.directory / f'member_{self.current_member-1:03d}', 'final', AirfRANSLinearNO)
            if any(self.member_dir.iterdir()):
                raise ValueError('unfinished member artifacts without a committed checkpoint')
            return 0, None
        _, manifest_path = inspect_checkpoint(self.member_dir, self.args.checkpoint, AirfRANSLinearNO)
        saved, weights = read_pair(manifest_path, AirfRANSLinearNO)
        if saved['model_spec'] != self.model_spec or saved['profile_spec'] != self.config:
            raise ValueError('AirfRANS checkpoint structure/profile mismatch')
        if saved['data_spec'] != self.data or saved['normalizer_spec'] != self.normalizers:
            raise ValueError('AirfRANS checkpoint data/normalizer mismatch')
        strict_load(model, weights)
        state = unpack_state(saved['resume_state']['optimizer'])
        optimizer.load_state_dict(state)
        scheduler.load_state_dict(unpack_state(saved['resume_state']['scheduler']))
        restore_random_state(saved['resume_state'], {})
        history = unpack_state(saved['resume_state']['sampler_state']).get('history')
        if history is None:
            # Narrow compatibility with L6 archives created before history was
            # bound to the committed epoch. Never consume later-epoch curves.
            history_path = self.member_dir / 'history.json'
            history = json.loads(history_path.read_text()) if history_path.is_file() else None
            if history is None or any(len(curve) != saved['resume_state']['epoch'] for curve in history['curves'][:4]):
                raise ValueError('legacy AirfRANS checkpoint lacks matching epoch history')
        self.completed_epoch = saved['resume_state']['epoch']
        expected_step = self.completed_epoch * self.steps_per_epoch
        saved_scheduler = unpack_state(saved['resume_state']['scheduler'])
        if saved['resume_state']['global_step'] != expected_step:
            raise ValueError('AirfRANS checkpoint global_step does not match resolved loader steps')
        if saved_scheduler.get('last_epoch') != expected_step:
            raise ValueError('AirfRANS checkpoint scheduler progress mismatch')
        return self.completed_epoch, history

    def complete_epoch(self, epoch, model, history):
        if self.steps_per_epoch is None:
            raise RuntimeError('prepare must bind optimizer and scheduler before checkpointing')
        if epoch != self.completed_epoch + 1:
            raise ValueError(f'AirfRANS checkpoint epoch order differs: expected {self.completed_epoch + 1}, got {epoch}')
        self.completed_epoch = epoch
        self._member_metadata = self._metadata(resume_state(
            self.optimizer, self.scheduler, epoch, self.steps_per_epoch,
            self.config['values']['training']['epochs'], {}, {'history': _json_history(history)}))
        save_pair(self.member_dir, model, self._member_metadata, AirfRANSLinearNO)
        serial = _json_history(history)
        (self.member_dir / 'history.json').write_text(json.dumps(serial, allow_nan=False))

    def bind(self, optimizer, scheduler):
        self.optimizer, self.scheduler = optimizer, scheduler

    def load_models(self):
        ensemble_path = self.directory / 'ensemble.json'
        if not ensemble_path.is_file():
            raise ValueError('missing LinearNO ensemble manifest')
        ensemble = json.loads(ensemble_path.read_text())
        members = ensemble.get('members') if isinstance(ensemble, dict) else None
        if (ensemble.get('family') != 'linearno' or ensemble.get('checkpoint_role') != 'final'
                or not isinstance(members, list)
                or len(members) != self.config['values']['training']['nmodel']):
            raise ValueError('LinearNO ensemble manifest family/count mismatch')
        for index, row in enumerate(members):
            if (row.get('member_id') != f'member_{index:03d}' or row.get('order') != index
                    or row.get('format') != 'state_dict' or not isinstance(row.get('path'), str)):
                raise ValueError('LinearNO ensemble manifest member order mismatch')
            path = (self.directory / row['path']).resolve()
            if path.parent != self.directory / f'member_{index:03d}' / 'weights' or not path.is_file():
                raise ValueError('LinearNO ensemble manifest member path mismatch')
            if row.get('sha256') != _sha256(path):
                raise ValueError('LinearNO ensemble manifest member checksum mismatch')
        models = []
        for index in range(self.config['values']['training']['nmodel']):
            self.current_member = index
            member = self.member_dir
            _, manifest_path = inspect_checkpoint(member, self.args.checkpoint, AirfRANSLinearNO)
            pair = json.loads(manifest_path.read_text())
            if (self.args.checkpoint == 'final' and
                    (member / pair['weights']['path']).resolve() != (self.directory / members[index]['path']).resolve()):
                raise ValueError('ensemble manifest does not identify the selected final weights')
            metadata, weights = read_pair(manifest_path, AirfRANSLinearNO)
            check_structure(metadata)
            if (metadata['model_spec'] != self.model_spec or
                    metadata['profile_spec'] != self.config or
                    metadata['data_spec'] != self.data or
                    metadata['normalizer_spec'] != self.normalizers):
                raise ValueError('LinearNO checkpoint structure/profile mismatch')
            model = AirfRANSLinearNO(**_constructor_kwargs(self.config))
            strict_load(model, weights)
            models.append(model.to(self.args.device).eval())
        if not self.evaluation:
            self._write_ensemble_manifest(models)
        return models

    def _write_ensemble_manifest(self, models):
        manifest = []
        for index in range(len(models)):
            member = self.directory / f'member_{index:03d}'
            pointer = member / 'checkpoints' / 'final.json'
            if not pointer.is_file():
                raise ValueError(f'missing final checkpoint for ensemble member {index}')
            pointer_data = json.loads(pointer.read_text())
            manifest_data = json.loads((member / 'checkpoints' / pointer_data['manifest']).read_text())
            weights = (member / manifest_data['weights']['path']).resolve()
            if weights.parent != member / 'weights':
                raise ValueError('ensemble state_dict must remain under the member weights directory')
            manifest.append(dict(member_id=f'member_{index:03d}', order=index,
                                 path=str(weights.relative_to(self.directory)),
                                 sha256=_sha256(weights), format='state_dict'))
        (self.directory / 'ensemble.json').write_text(json.dumps(
            dict(family='linearno', checkpoint_role='final', members=manifest), indent=2) + '\n')

    def finish_ensemble(self, models):
        states = [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in models]
        torch.save(states, self.directory / 'ensemble_state_dict.pth')
        self._write_ensemble_manifest(models)

    def eval_dir(self):
        if self.evaluation and self.recorder is not None:
            return Path(self.recorder.result_dir)
        path = self.directory / 'evaluations' / ('eval_' + uuid4().hex[:12])
        path.mkdir(parents=True, exist_ok=False)
        return path


def _json_history(history):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value
    return convert(history)


def _restore_coef_norm(metadata):
    if metadata is None:
        return None
    record = metadata['normalizer_spec']['records'].get('coef_norm')
    if record is None:
        raise ValueError('AirfRANS metadata has no saved coef_norm')
    states = record['states']
    expected = ('input_mean', 'input_std', 'output_mean', 'output_std')
    if set(states) != set(expected):
        raise ValueError('AirfRANS coef_norm metadata fields mismatch')
    return tuple((value.numpy() if isinstance(value, torch.Tensor) else value)
                 for value in (restore_numerical_state(states[name]) for name in expected))


def _load_dataset(data_dir, args, *, train=True, coef_norm=None):
    from dataset.dataset import Dataset
    with (Path(data_dir) / 'manifest.json').open() as stream:
        manifest = json.load(stream)
    train_names = manifest[args.task + '_train']
    n = int(.1 * len(train_names))
    if n:
        train_names, val_names = train_names[:-n], train_names[-n:]
    else:
        val_names = []
    if train:
        if coef_norm is None:
            train_dataset, coef_norm = Dataset(train_names, norm=True, sample=None, my_path=str(data_dir))
        else:
            train_dataset = Dataset(train_names, sample=None, coef_norm=coef_norm, my_path=str(data_dir))
        val_dataset = Dataset(val_names, sample=None, coef_norm=coef_norm, my_path=str(data_dir))
        return manifest, train_dataset, val_dataset, coef_norm
    if coef_norm is None:
        raise ValueError('AirfRANS evaluation requires saved train-fit coef_norm')
    return manifest, coef_norm


def run_cli(args):
    if getattr(args, 'linearno_family', None) == 'linearno_loop':
        from cdlno.linearno_loop.air_entry import run_cli as loop_run_cli
        return loop_run_cli(args)
    """Run the original AirfRANS dataset/training/eval actions for LinearNO."""
    from cdlno.experiment import finish, start
    import train as air_train
    import utils.metrics as metrics

    config = args._linearno_config
    hparams = _hparams(config)
    recorder = start(args, 'airfrans', evaluation=bool(args.eval), hparams=hparams)
    data_dir = _data_root(args.my_path, evaluation=bool(args.eval))
    saved_coef_norm = _restore_coef_norm(getattr(args, '_linearno_metadata', None))
    if args.eval:
        manifest, coef_norm = _load_dataset(data_dir, args, train=False, coef_norm=saved_coef_norm)
    else:
        manifest, train_dataset, val_dataset, coef_norm = _load_dataset(
            data_dir, args, train=True, coef_norm=saved_coef_norm)
    if not args.eval:
        hparams['total_steps'] = ((len(train_dataset) // hparams['batch_size']) + 1) * hparams['nb_epochs']
    args._linearno_normalizers = {'coef_norm': coef_norm}
    run = AirRun(args, config, data_dir, coef_norm, recorder, manifest=manifest,
                 evaluation=bool(args.eval))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError(f'requested {device}, but CUDA is unavailable')
    if args.eval:
        models = run.load_models()
        result_dir = run.eval_dir()
        scores = metrics.Results_test(str(device), [models], [hparams], coef_norm, str(data_dir),
                                      str(result_dir), n_test=3, criterion='MSE',
                                      s=args.task + '_test' if args.task != 'scarce' else 'full_test')
        pressure = _pressure_rL2_dataset(str(device), models, hparams, coef_norm,
                                         data_dir, manifest[args.task + '_test'] if args.task != 'scarce'
                                         else manifest['full_test'])
        (result_dir / 'pressure_rL2.json').write_text(json.dumps(pressure, indent=2) + '\n')
        np.save(result_dir / 'true_coefs', scores[0]); np.save(result_dir / 'pred_coefs_mean', scores[1])
        np.save(result_dir / 'pred_coefs_std', scores[2])
        recorder.record_metrics(dict(result_dir=str(result_dir), field_metric='normalized four-channel MSE',
                                     pressure_field_metric='physical pressure-channel rL2 by volume/surface region',
                                     pressure_rL2=pressure,
                                     force_metric='relative coefficient error and Spearman', checkpoint=args.checkpoint))
        finish(args)
        return result_dir

    models = []
    for index in range(config['values']['training']['nmodel']):
        run.current_member = index
        model = AirfRANSLinearNO(**_constructor_kwargs(config)).to(device)
        member_pointer = run.member_dir / 'checkpoints' / 'latest.json'
        if args.resume and member_pointer.exists():
            saved, path = inspect_checkpoint(run.member_dir, args.checkpoint, AirfRANSLinearNO)
            check_structure(saved)
            if any(saved[k] != value for k, value in (
                    ('model_spec', run.model_spec), ('profile_spec', config),
                    ('data_spec', run.data), ('normalizer_spec', run.normalizers))):
                raise ValueError('AirfRANS resume member metadata mismatch')
            if saved['resume_state']['checkpoint_role'] == 'final':
                _, weights = read_pair(path, AirfRANSLinearNO)
                strict_load(model, weights)
                restore_random_state(saved['resume_state'], {})
                if not (run.member_dir / 'model').exists():
                    torch.save(model, run.member_dir / 'model')
                models.append(model)
                continue
        recorder.attach_model(model, hparams=hparams, member=index, protocol=dict(
            family='linearno', profile=config['profile'], objective=config['values']['objective'],
            steps_per_epoch=len(train_dataset) // hparams['batch_size'] + 1,
            total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs']))
        from .air_visualization import AirFields
        if not hasattr(recorder, '_field_visualizers'):
            recorder._field_visualizers = {}
        recorder._field_visualizers[index] = AirFields(run.directory, 'airfrans', 'LinearNO',
                                                       seed=args.seed, member=index)
        # The original train function constructs Adam/OneCycle and preserves its
        # random sampling/validation cadence.  The adapter observes each epoch.
        model = air_train.main(str(device), train_dataset, val_dataset, model, hparams,
                               str(run.member_dir), criterion='MSE_weighted', reg=args.weight,
                               val_iter=10, name_mod='LinearNO', val_sample=True,
                               record=recorder, record_member=index, visualization_norm=coef_norm,
                               linearno_run=run)
        # Native final plots/pickle are observations, not the next member's RNG.
        saved, _ = inspect_checkpoint(run.member_dir, 'final', AirfRANSLinearNO)
        restore_random_state(saved['resume_state'], {})
        models.append(model)
    run.finish_ensemble(models)
    finish(args)
    return run.directory


def pressure_relative_l2(prediction, target, surf, coef_norm):
    """Physical pressure rL2 for one graph, with no stabilizing epsilon."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError('AirfRANS pressure rL2 requires prediction/target [N,4]')
    if surf.dtype is not torch.bool or surf.shape != (prediction.shape[0],):
        raise ValueError('AirfRANS pressure rL2 requires boolean surf [N]')
    physical_pred = prediction * torch.as_tensor(coef_norm[3], device=prediction.device,
                                                  dtype=prediction.dtype) + torch.as_tensor(coef_norm[2], device=prediction.device,
                                                                                           dtype=prediction.dtype)
    physical_target = target * torch.as_tensor(coef_norm[3], device=target.device,
                                                dtype=target.dtype) + torch.as_tensor(coef_norm[2], device=target.device,
                                                                                       dtype=target.dtype)
    values = {}
    for name, mask in (('volume', ~surf), ('surface', surf)):
        if not bool(mask.any()):
            raise ValueError(f'AirfRANS pressure rL2 {name} region is empty')
        truth = physical_target[mask, 2].reshape(-1)
        denominator = torch.linalg.vector_norm(truth)
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError(f'AirfRANS pressure rL2 {name} target norm must be finite and nonzero')
        values[name] = (torch.linalg.vector_norm(physical_pred[mask, 2].reshape(-1) - truth) /
                        denominator)
    return values


def _pressure_rL2_dataset(device, models, hparams, coef_norm, data_dir, names):
    """Additional paper metric; original Results_test remains authoritative."""
    from dataset.dataset import Dataset
    from torch_geometric.loader import DataLoader
    import utils.metrics as metrics
    # ``Infer_test`` indexes hyperparameters by ensemble member.  Normalize a
    # native single-member mapping here so the auxiliary metric is independent
    # of the training objective and works for both one and many members.
    if isinstance(hparams, dict):
        hparams = [hparams] * len(models)
    dataset = Dataset(names, sample=None, coef_norm=coef_norm, my_path=str(data_dir))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    sums = [dict(volume=0.0, surface=0.0) for _ in models]
    count = 0
    for data in loader:
        outputs = []
        for index, model in enumerate(models):
            # Keep each member in its own Infer_test call.  The historical
            # helper uses list multiplication for its zero buffers, which is
            # harmless for one model but aliases buffers for an ensemble.
            output, _ = metrics.Infer_test(device, [model], [hparams[index]], data,
                                           coef_norm=coef_norm)
            outputs.append(output[0])
        for index, output in enumerate(outputs):
            values = pressure_relative_l2(output, data.y, data.surf, coef_norm)
            sums[index]['volume'] += float(values['volume'])
            sums[index]['surface'] += float(values['surface'])
        count += 1
    if count < 1:
        raise ValueError('AirfRANS pressure rL2 requires a nonempty test set')
    return [{key: value / count for key, value in row.items()} for row in sums]
