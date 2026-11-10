"""Native six-Standard-task adapter. No exp imports or data reads on import.

The original entry owns all data, forward/loss, optimizer and scheduler loops.
This adapter owns only LinearNO resolution, records, normalizer state and strict
epoch-boundary continuation. Old families never receive these kwargs or flags.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import re
import sys
from uuid import uuid4

from .profiles import DEFAULT_PROFILE, PROFILES, resolve_config, digest
from .schema import (make_metadata, read_metadata, write_metadata, normalizer_record,
                     restore_numerical_state, unpack_state, validate_model_spec)

ROOT = Path(__file__).resolve().parents[2]
KEYS = ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh')
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity')
ARG_FIELDS = dict(n_hidden='model.hidden', n_layers='model.layers', n_heads='model.heads',
    mlp_ratio='model.ffn_ratio', dropout='model.dropout', ref='model.ref', unified_pos='model.unified_pos',
    linearno_variant='model.linearno_variant', linearno_rank='model.linearno_rank',
    lr='training.lr', epochs='training.epochs', weight_decay='training.weight_decay',
    batch_size='training.batch_size', max_grad_norm='training.gradient_clip', seed='runtime.seed',
    save_name='runtime.save_name')


def model_class():
    from model.LinearNO import Model
    return Model


def constructor_kwargs(config):
    m = config['values']['model']
    names = dict(space_dim='space_dim', n_layers='layers', n_hidden='hidden', n_head='heads',
        dropout='dropout', Time_Input='time_input', act='activation', mlp_ratio='ffn_ratio',
        fun_dim='fun_dim', out_dim='out_dim', ref='ref', unified_pos='unified_pos', H='H', W='W',
        linearno_variant='linearno_variant', linearno_rank='linearno_rank')
    return {k: m[v] for k,v in names.items()}


def _structural_check(metadata, task):
    if metadata['profile_spec']['task'] != task:
        raise ValueError('checkpoint task mismatch')
    expected = dict(class_path='model.LinearNO.Model', constructor_kwargs=constructor_kwargs(metadata['profile_spec']))
    if metadata['model_spec'] != expected:
        raise ValueError('checkpoint model_spec/profile structural mismatch')
    validate_model_spec(expected, model_class())


def parse_args(parser, task, tokens):
    if task not in TASKS:
        parser.error('LinearNO Standard adapter supports only the six Standard tasks')
    parser.add_argument('--linearno-profile', choices=PROFILES, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-variant', choices=('plain','temp','conv','conv_temp'), default=argparse.SUPPRESS)
    parser.add_argument('--linearno-rank', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--experiment-dir', '--linearno-run-dir', dest='linearno_run_dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', default=None, help='final, latest or epoch_XXXX within the explicit run')
    args = parser.parse_args(tokens)
    probe = copy.deepcopy(parser); probe._defaults.clear()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    explicit = vars(probe.parse_args(tokens))
    try:
        expected_key = KEYS[1] if task == 'elasticity' else KEYS[0]
        if args.model != expected_key:
            raise ValueError(f'{task} requires --model {expected_key}')
        if 'slice_num' in explicit:
            raise ValueError('LinearNO rank requires --linearno-rank, not --slice_num')
        if args.eval not in (0,1) or (args.eval and args.resume):
            raise ValueError('choose exactly train, resume or eval')
        if args.checkpoint and not (args.eval or args.resume):
            raise ValueError('--checkpoint requires eval or resume')
        for name, expected in dict(downsample=1 if task == 'ns' else 5, downsamplex=1, downsampley=1).items():
            if hasattr(args,name) and getattr(args,name) != expected:
                raise ValueError(f'LinearNO task grid/sampling requires {name}={expected}')
        overrides = {path: explicit[name] for name,path in ARG_FIELDS.items() if name in explicit}
        if 'model.unified_pos' in overrides:
            if overrides['model.unified_pos'] not in (0,1):
                raise ValueError('unified_pos must be 0 or 1')
            overrides['model.unified_pos'] = bool(overrides['model.unified_pos'])
        profile = getattr(args,'linearno_profile',DEFAULT_PROFILE)
        if args.eval or args.resume:
            from .checkpoint import inspect_checkpoint
            if args.linearno_run_dir is None:
                raise ValueError('eval/resume requires an explicit --experiment-dir')
            selector = args.checkpoint or ('latest' if args.resume else 'final')
            metadata, checkpoint = inspect_checkpoint(args.linearno_run_dir, selector, model_class())
            _structural_check(metadata, task)
            initial = read_metadata(args.linearno_run_dir/'architecture.json', constructor=model_class())
            if any(initial[k] != metadata[k] for k in ('model_spec','profile_spec','data_spec','normalizer_spec','provenance_spec')):
                raise ValueError('checkpoint differs from immutable run metadata')
            config = metadata['profile_spec']
            if 'linearno_profile' in explicit and profile != config['profile']:
                raise ValueError('explicit profile conflicts with checkpoint')
            for path, value in overrides.items():
                section, field = path.split('.')
                if section != 'runtime' and config['values'][section][field] != value:
                    raise ValueError(f'explicit {path} conflicts with checkpoint')
                if args.resume and section == 'runtime' and config['values'][section][field] != value:
                    raise ValueError(f'resume {path} conflicts with checkpoint')
            if args.resume and metadata['resume_state']['epoch'] >= config['values']['training']['epochs']:
                raise ValueError('run has completed its resolved epochs')
            args._linearno_metadata, args._linearno_checkpoint = metadata, checkpoint
        else:
            contract = 'standard_temporal_l5' if task in ('ns', 'plasticity') else 'standard_static_l4'
            config = resolve_config(task, profile, explicit=overrides, contract=contract)
        if task == 'darcy' and config['profile'] == 'official_release' and config['values']['training']['epochs'] != 500:
            raise ValueError('Darcy official_release fixes scheduler epochs=500; non-500 training epochs are rejected')
        args.linearno_family, args.linearno_task = 'linearno', task
        args._linearno_config = config
        args._linearno_model_spec = dict(class_path='model.LinearNO.Model',constructor_kwargs=constructor_kwargs(config))
        args._linearno_normalizers = {}
        for name,path in ARG_FIELDS.items():
            section,field = path.split('.')
            setattr(args,name,config['values'][section][field])
        args.linearno_profile = config['profile']
        if args.save_name is None:
            args.save_name = f'{task}_linearno_{config["profile"]}_{args.linearno_variant}_M{args.linearno_rank}_seed{args.seed}'
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.save_name) or args.save_name in ('.','..'):
            raise ValueError('save_name must be a filename stem')
        if args.linearno_run_dir is None:
            from cdlno.experiment import timestamp
            base = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT/'output'))
            args.linearno_run_dir = base/task/'linearno'/config['profile']/f'{args.linearno_variant}_M{args.linearno_rank}'/f'seed{args.seed}_{timestamp()}_{uuid4().hex[:8]}'
        args.linearno_run_dir = args.linearno_run_dir.resolve()
        return args
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


def model_kwargs(args, **grid):
    # The six task entries share this interface for LinearNO and PhyLoopNO.
    if hasattr(args, '_linearno_loop_config'):
        from linearno_loop.versioning import is_v5
        if is_v5(args._linearno_loop_config):
            from cdlno.linearno_loop.standard_entry import model_kwargs as loop_kwargs
            return loop_kwargs(args, **grid)
    kwargs = dict(args._linearno_model_spec['constructor_kwargs'])
    for key, value in grid.items():
        if kwargs[key] != value:
            raise ValueError(f'data grid {key}={value} conflicts with model metadata {kwargs[key]}')
    return kwargs


def start(args, task):
    import numpy as np
    import torch
    from cdlno.experiment import start as begin
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    return begin(args,task,evaluation=bool(args.eval))


def finish(args):
    from cdlno.experiment import finish as end
    end(args)


def verify_data(args):
    from .checkpoint import sha256
    values = args._linearno_config['values']
    files = {name.split(':')[0]: sha256(Path(args.data_path) if args.linearno_task == 'plasticity'
                                      else Path(args.data_path)/name.split(':')[0]) for name in values['data']['files']}
    checksums = dict(dataset_manifest=digest(files), **files)
    data = dict(split=values['data']['split'], sampling=values['data']['sampling'], checksums=checksums,
                scope='actual files read by unchanged hxh task loader',
                protocol_decisions=['L4 user: batch-mean rL2, released batch-sum objective is scaled by 1/B',
                                    'hxh validation every epoch; Pipe retains first1200 selection'])
    if args.linearno_task in ('ns', 'plasticity'):
        data['protocol_decisions'] = ['hxh temporal batch-sum objective and time reduction unchanged',
            'NS: ten truth-fed train calls, one optimizer/scheduler; ten prediction-fed eval calls',
            'Plasticity: per-sample torch.randperm20, twenty optimizers per outer scheduler; original point order']
    if hasattr(args,'_linearno_metadata'):
        for key in ('split','sampling','checksums'):
            if data[key] != args._linearno_metadata['data_spec'][key]:
                raise ValueError(f'data {key} differs from saved run')
    args._linearno_data = data


def normalizer(args, name, training_tensor, cls):
    if args.eval or args.resume:
        records = args._linearno_metadata['normalizer_spec']['records']
        if name not in records or records[name]['algorithm'] != 'UnitTransformer: mean(dim=(0,1)); std(dim=(0,1))+1e-8':
            raise ValueError(f'missing/incompatible saved normalizer {name}')
        value = cls.__new__(cls)  # no fit, no constructor call on eval/resume
        states = records[name]['states']
        if set(states) != {'mean','std'}:
            raise ValueError('normalizer state fields mismatch')
        value.mean = restore_numerical_state(states['mean'])
        value.std = restore_numerical_state(states['std'])
    else:
        value = cls(training_tensor)
    args._linearno_normalizers[name] = value
    return value


def provenance():
    from cdlno.provenance import provenance as current_provenance
    return current_provenance()


class StandardRun:
    def __init__(self, args, model):
        from cdlno.experiment import session
        self.args, self.model = args, model
        self.directory = args.linearno_run_dir
        self.recorder = session(args)
        if self.recorder is None:
            raise ValueError('start the native experiment before constructing StandardRun')
        self.result_dir = self.recorder.result_dir
        self.recorder.attach_model(model)
        self.completed_epoch = 0
        self.start_epoch = 0

    def prepare(self, optimizer, scheduler, train_loader, test_loader):
        from .checkpoint import resume_state, read_pair, strict_load, restore_random_state
        from cdlno.training_state import _optimizer_signature
        self.optimizer, self.scheduler = optimizer, scheduler
        self.steps = len(train_loader)
        self.updates_per_batch = 20 if self.args.linearno_task == 'plasticity' else 1
        self.optimizer_steps = self.steps * self.updates_per_batch
        self.generators = {name: loader.generator for name,loader in (('train',train_loader),('test',test_loader)) if loader.generator is not None}
        self.sampler = {name: dict(type=type(loader.sampler).__name__, epoch_boundary=True,
                                  replacement=getattr(loader.sampler,'replacement',False))
                        for name,loader in (('train',train_loader),('test',test_loader))}
        args = self.args; cfg = args._linearno_config
        self.scheduler_plan = dict(type=type(scheduler).__name__, epochs=args.epochs,
            steps_per_epoch=self.steps if type(scheduler).__name__=='OneCycleLR' else 1,
            total_steps=getattr(scheduler,'total_steps',args.epochs), state_at_construction=scheduler.state_dict())
        records = {name: normalizer_record(dict(mean=obj.mean,std=obj.std), fit_split='train only',
            data_checksum=next(iter(args._linearno_data['checksums'].values())),
            algorithm='UnitTransformer: mean(dim=(0,1)); std(dim=(0,1))+1e-8') for name,obj in args._linearno_normalizers.items()}
        normalizers = dict(policy='saved_train_fit' if records else 'none',records=records)
        data = dict(args._linearno_data, ntrain=len(train_loader.dataset), ntest=len(test_loader.dataset),
                    scheduler=self.scheduler_plan, train_batch_size=train_loader.batch_size,
                    test_batch_size=test_loader.batch_size, optimizer_signature=_optimizer_signature(self.model,optimizer))
        data = json.loads(json.dumps(data))  # canonical JSON tuples/lists before immutable comparisons
        if args.linearno_task in ('ns', 'plasticity'):
            data['temporal_progress'] = dict(optimizer_updates_per_batch=self.updates_per_batch,
                scheduler_updates_per_batch=1, forwards_per_train_batch=20 if args.linearno_task == 'plasticity' else 10,
                time_permutation='per-sample torch.randperm20' if args.linearno_task == 'plasticity' else None,
                global_step_unit='optimizer updates')
        prov = provenance()
        if args.eval or args.resume:
            saved, weights = read_pair(args._linearno_checkpoint, model_class())
            for key,actual in (('data_spec',data),('normalizer_spec',normalizers)):
                if saved[key] != actual:
                    raise ValueError(f'{key} differs from saved run')
            if args.resume:
                if saved['provenance_spec']['source_sha256'] != prov['source_sha256']:
                    raise ValueError('resume source code differs from saved run')
                epoch = saved['resume_state']['epoch']
                # Reject rollback even if a newer pair committed before latest was published.
                committed = [p for p in (self.directory/'checkpoints').glob('epoch_*.json') if '.metadata.' not in p.name]
                if epoch != max(int(p.stem.split('_')[1]) for p in committed):
                    raise ValueError('resume requires the newest committed epoch in this run')
                state = saved['resume_state']
                expected_steps = epoch*self.scheduler_plan['steps_per_epoch']
                if state['global_step'] != epoch*self.optimizer_steps or unpack_state(state['scheduler'])['last_epoch'] != expected_steps:
                    raise ValueError('resume optimizer/scheduler progress mismatch')
                if unpack_state(state['sampler_state']) != self.sampler:
                    raise ValueError('sampler state mismatch')
                optim_state = unpack_state(state['optimizer'])
                scheduler_state = unpack_state(state['scheduler'])
                if scheduler_state.get('_step_count') != expected_steps+1:
                    raise ValueError('resume scheduler step counter mismatch')
                for group, current in zip(optim_state['param_groups'],optimizer.param_groups):
                    if len(group['params']) != len(current['params']):
                        raise ValueError('resume optimizer group shape mismatch')
                    for index,p in zip(group['params'],current['params']):
                        for name,value in optim_state['state'].get(index,{}).items():
                            if name in ('exp_avg','exp_avg_sq','max_exp_avg_sq') and (value.shape!=p.shape or value.dtype!=p.dtype):
                                raise ValueError('resume optimizer moment shape/dtype mismatch')
                strict_load(self.model,weights)
                optimizer.load_state_dict(optim_state)
                scheduler.load_state_dict(scheduler_state)
                self.start_epoch = self.completed_epoch = epoch
                restore_random_state(state,self.generators)  # restore LAST, after construction/loading
            self.metadata = saved
        else:
            self.metadata = make_metadata(constructor=model_class(),profile_spec=cfg,
                model_spec=args._linearno_model_spec,data_spec=data,
                objective_spec=cfg['values']['objective'],evaluation_spec=cfg['values']['evaluation'],
                provenance_spec=prov,normalizer_spec=normalizers,
                resume_state=resume_state(optimizer,scheduler,0,self.optimizer_steps,args.epochs,self.generators,self.sampler),
                ensemble_manifest=[])
            write_metadata(self.directory/'architecture.json',self.metadata,constructor=model_class())
        self.recorder.update_protocol(dict(linearno_data=data, scheduler_resolved=self.scheduler_plan))
        if hasattr(args,'_linearno_metadata'):
            del args._linearno_metadata  # do not dump optimizer/RNG base64 via the legacy print(args)

    def complete_epoch(self, epoch):
        if epoch != self.completed_epoch+1:
            raise ValueError('epoch order differs from resume progress')
        self.completed_epoch = epoch

    def load(self, model):
        from .checkpoint import read_pair, strict_load
        _, weights = read_pair(self.args._linearno_checkpoint,model_class())
        strict_load(model,weights)

    def save(self, model):
        from .checkpoint import resume_state, save_pair
        if self.args.eval:
            raise ValueError('evaluation cannot save training weights')
        if self.scheduler.last_epoch != self.completed_epoch*self.scheduler_plan['steps_per_epoch']:
            raise ValueError('actual scheduler cadence differs from metadata')
        metadata = copy.deepcopy(self.metadata)
        metadata['resume_state'] = resume_state(self.optimizer,self.scheduler,self.completed_epoch,
            self.optimizer_steps,self.args.epochs,self.generators,self.sampler)
        metadata['metadata_hash'] = digest({k:v for k,v in metadata.items() if k!='metadata_hash'})
        return save_pair(self.directory,model,metadata,model_class())
