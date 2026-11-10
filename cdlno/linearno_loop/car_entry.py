"""ShapeNet loop adapter; native LinearNO train/objective/field/drag functions."""
import numpy as np
import torch
import copy
import json
from pathlib import Path
from cdlno.linearno import car_entry as pure
from cdlno.linearno.checkpoint import resume_state, strict_load
from cdlno.training_state import _optimizer_signature
from linearno_loop.versioning import make_metadata, write_metadata
from linearno_loop.contracts import require_equal
from .industrial_state import construct, generators, data_contract, legacy_data_view, restore_training, export_state
from .versioning import checkpoint_api, provenance
restore_coef_norm = pure.restore_coef_norm
load_data = pure.load_data
evaluate = pure.evaluate

def inspect_data(args):
    return pure.inspect_data(legacy_data_view(args))

class CarRun:
    objective = pure.CarRun.objective

    def _resume_state(self, optimizer, scheduler, epoch):
        return self.checkpoint.resume_state(optimizer, scheduler, epoch, self.steps, self.args.nb_epochs, self.generators, dict(train='RandomSampler, drop_last=True, independent generator', test='SequentialSampler', epoch_boundary=True))

    def __init__(self, args, data_spec, coef_norm, recorder=None):
        self.args, self.config, self.directory = (args, args._linearno_config, Path(args.linearno_run_dir))
        self.data, self.coef, self.recorder = (copy.deepcopy(data_spec), coef_norm, recorder)
        self.completed_epoch = 0
        self.generators = generators(args)
        self.checkpoint = checkpoint_api(args._linearno_loop_config)
        self.current_provenance = provenance(args._linearno_loop_config, task='car')
        self.provenance = self.current_provenance
        from cdlno.linearno.schema import normalizer_record
        from cdlno.linearno.profiles import digest
        checksum = digest(self.data['checksums'])
        self.data['checksums']['normalizer_fit_dataset'] = checksum
        self.normalizers = dict(policy='saved_train_fit', records=dict(coef_norm=normalizer_record(dict(zip(('input_mean', 'input_std', 'output_mean', 'output_std'), coef_norm)), fit_split='train_samples only', data_checksum=checksum, algorithm='native all-point streaming mean/std; encode=(x-mean)/(std+1e-8); decode=x*std+mean')))
        tr = self.config['values']['training']
        self.steps = self.data['ntrain'] // tr['batch_size']
        self.data['scheduler'] = dict(epochs=tr['epochs'], actual_steps_per_epoch=self.steps, total_steps=(self.steps + 1) * tr['epochs'], final_div_factor=1000, drop_last=True)
        if hasattr(args, '_linearno_metadata'):
            saved = args._linearno_metadata
            if args.resume and saved['provenance_spec']['source_sha256'] != self.current_provenance['source_sha256']:
                raise ValueError('resume source differs')
            self.provenance = saved['provenance_spec']
            native = copy.deepcopy(saved['data_spec']['runtime'])
            native.pop('optimizer_signature')
            require_equal(data_contract(args._linearno_loop_config, self.data), {**saved['data_spec'], 'runtime': native}, 'saved.data_spec')
            require_equal(self.normalizers, saved['normalizer_spec'], 'saved.normalizer_spec')
            self.data['optimizer_signature'] = saved['data_spec']['runtime']['optimizer_signature']

    def metadata(self, optimizer, scheduler, epoch, model=None):
        if model is None:
            raise ValueError('metadata requires a constructed model')
        return make_metadata(self.args._linearno_loop_config, data_spec=data_contract(self.args._linearno_loop_config, self.data), normalizer_spec=self.normalizers, provenance_spec=self.provenance, resume_state=self._resume_state(optimizer, scheduler, epoch), ensemble_manifest=[], parameter_measurement=self.checkpoint.measure_parameters(model, self.args._linearno_loop_config))

    def prepare(self, model, optimizer, scheduler):
        self.optimizer, self.scheduler = (optimizer, scheduler)
        if scheduler.total_steps != self.data['scheduler']['total_steps']:
            raise ValueError('Car scheduler differs from release')
        signature = json.loads(json.dumps(_optimizer_signature(model, optimizer)))
        if 'optimizer_signature' in self.data:
            require_equal(self.data['optimizer_signature'], signature, 'optimizer_signature')
        self.data['optimizer_signature'] = signature
        if self.recorder:
            self.recorder.record_training_setup(optimizer, scheduler)
        if self.args.resume:
            saved, _ = self.checkpoint.read_pair(self.args._linearno_checkpoint, expected=self.args._linearno_loop_config)
            self.completed_epoch = restore_training(saved, self.args._linearno_checkpoint, model, optimizer, scheduler, steps=self.steps, generators=self.generators, current_provenance=self.current_provenance)
        else:
            write_metadata(self.directory / 'architecture.json', self.metadata(optimizer, scheduler, 0, model))
        return self.completed_epoch

    def complete_epoch(self, epoch, model, metrics):
        if epoch != self.completed_epoch + 1 or self.scheduler.last_epoch != epoch * self.steps:
            raise ValueError('Car epoch/scheduler mismatch')
        if self.recorder:
            self.recorder.record_epoch(epoch, metrics)
        self.checkpoint.save_pair(self.directory, model, self.metadata(self.optimizer, self.scheduler, epoch, model))
        self.completed_epoch = epoch

    def export_final(self, model):
        export_state(model, self.directory / f'model_{self.args.nb_epochs}.pth')

    def load(self):
        saved, state = self.checkpoint.read_pair(self.args._linearno_checkpoint, expected=self.args._linearno_loop_config)
        require_equal(saved['data_spec'], data_contract(self.args._linearno_loop_config, self.data), 'data_spec')
        require_equal(saved['normalizer_spec'], self.normalizers, 'normalizer_spec')
        model = construct(self.args)
        strict_load(model, state)
        return model.to(self.args.device).eval()

    def train(self, device, train_dataset, val_dataset, model, hparams, path, reg, val_iter, coef_norm, record):
        """Called by native train.main only with an explicit LinearNO run."""
        from torch_geometric.loader import DataLoader
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['lr'], total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'], final_div_factor=1000.0)
        first = self.prepare(model, optimizer, scheduler)
        for epoch in range(first, hparams['nb_epochs']):
            model.train()
            rows = []
            loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True, generator=self.generators['train'])
            for data, geom in loader:
                data, geom = (data.to(device), geom.to(device))
                optimizer.zero_grad()
                loss, pressure, velocity = self.objective(model((data, geom)), data)
                loss.backward()
                optimizer.step()
                scheduler.step()
                rows.append([loss.item(), pressure.item(), velocity.item()])
            total, pressure, velocity = np.mean(rows, axis=0).tolist()
            metrics = dict(total=total, pressure=pressure, velocity=velocity, objective=self.config['values']['objective'])
            if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
                model.eval()
                values = []
                with torch.no_grad():
                    for data, geom in DataLoader(val_dataset, batch_size=1, generator=self.generators['test']):
                        data, geom = (data.to(device), geom.to(device))
                        values.append([v.item() for v in self.objective(model((data, geom)), data)])
                metrics['validation_components'] = np.mean(values, axis=0).tolist()
            if record:
                record.visualize(model, epoch + 1, hparams['nb_epochs'], dataset=val_dataset, coef_norm=coef_norm)
            print('LinearNO Car epoch', epoch + 1, metrics)
            self.complete_epoch(epoch + 1, model, metrics)
        if hasattr(self, 'export_final'):
            self.export_final(model)
        else:
            torch.save(model, Path(path) / f"model_{hparams['nb_epochs']}.pth")
        return model

def run_cli(args):
    from cdlno.experiment import start, finish
    import train as native_train
    hparams = dict(lr=args.lr, batch_size=args.batch_size, nb_epochs=args.nb_epochs)
    recorder = start(args, 'car', evaluation=bool(args.eval), hparams=hparams)
    data_spec = inspect_data(args)
    train_dataset, test_dataset, coef = load_data(args, data_spec)
    run = CarRun(args, data_spec, coef, recorder)
    model = run.load() if args.eval else construct(args).to(args.device)
    recorder.attach_model(model, hparams=hparams, protocol=data_spec)
    if args.eval:
        result = evaluate(run, model, test_dataset)
    else:
        result = native_train.main(args.device, train_dataset, test_dataset, model, hparams, str(run.directory), reg=args.weight, val_iter=args.val_iter, coef_norm=coef, record=recorder, linearno_run=run)
    finish(args)
    return result
