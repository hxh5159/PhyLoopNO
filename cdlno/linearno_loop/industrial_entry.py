"""Loop-only industrial selection. No data access or model allocation while parsing."""
import argparse
import copy
import importlib
import os
from pathlib import Path
import random
from uuid import uuid4
from linearno_loop.contracts import HISTORY_FLAGS, PRESETS, TOPOLOGY_FIELDS, read_json
from linearno_loop.versioning import OPTIONS, read_metadata, resolve_config, restore_config, run_directory_id
from cdlno.linearno.profiles import PROFILES, DEFAULT_PROFILE
from .standard_entry import _loop_parser
ROOT = Path(__file__).resolve().parents[2]

def intercept(parser, tokens, *, task, evaluation, selected_model):
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument('--experiment-dir', '--linearno-run-dir', '--run_dir', dest='directory', type=Path)
    probe.add_argument('--resume', action='store_true')
    probe.add_argument('--eval', type=int, default=0)
    intent, _ = probe.parse_known_args(tokens)
    explicit, remaining = _loop_parser().parse_known_args(tokens)
    explicit = vars(explicit)
    try:
        path = intent.directory / 'architecture.json' if intent.directory else None
        saved = bool((evaluation or intent.eval or intent.resume) and path and path.is_file() and (read_json(path).get('family') == 'linearno_loop'))
        if not explicit and (not saved):
            return None
        flags = {'--' + name for name in HISTORY_FLAGS} | {'--' + name.replace('_', '-') for name in HISTORY_FLAGS} | {'--linearno-fair-run'}
        if any((t.split('=')[0] in flags for t in tokens)):
            raise ValueError('loop cannot mix A/K/history/fair-run flags, even false')
        if selected_model not in (None, 'LinearNO'):
            raise ValueError('loop requires the LinearNO model selection')
        key = '--model' if task == 'airfrans' else '--cfd_model'
        if selected_model is None:
            if not saved:
                raise ValueError('loop train requires explicit ' + key + ' LinearNO')
            remaining = [key, 'LinearNO', *remaining]
        if explicit.get('linearno_loop') == '0':
            if saved or set(explicit) != {'linearno_loop'}:
                raise ValueError('disabled loop conflicts with loop metadata/fields')
            pure = importlib.import_module('cdlno.linearno.' + ('air_entry' if task == 'airfrans' else 'car_entry'))
            return pure.parse_args(parser, remaining, evaluation=evaluation)
        if not (saved or evaluation or intent.eval or intent.resume) and explicit.get('linearno_loop') != '1':
            raise ValueError('loop train requires --linearno-loop 1')
        return parse_args(parser, remaining, explicit, task=task, evaluation=evaluation)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))

def parse_args(parser, tokens, loop_explicit, *, task, evaluation):
    parser.add_argument('--linearno-profile', choices=PROFILES, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-variant', choices=('airfrans',) if task == 'airfrans' else ('shapenet',), default=argparse.SUPPRESS)
    for field in ('rank', 'hidden', 'heads', 'ffn-ratio'):
        parser.add_argument('--linearno-' + field, type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-dropout', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--experiment-dir', '--linearno-run-dir', dest='linearno_run_dir', type=Path)
    parser.add_argument('--save-name', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default=None)
    if task == 'airfrans':
        parser.add_argument('--gpu', type=int, default=None)
        parser.add_argument('--linearno-ref', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--linearno-unified-pos', type=int, choices=(0, 1), default=argparse.SUPPRESS)
        parser.add_argument('--subsampling', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--r', type=float, default=argparse.SUPPRESS)
        parser.add_argument('--max-neighbors', '--max_neighbors', dest='max_neighbors', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--debug', type=int, choices=(0,), default=argparse.SUPPRESS)
        parser.add_argument('--epochs', dest='nb_epochs', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--eval', type=int, choices=(0, 1), default=0)
    args = parser.parse_args(tokens)
    probe = copy.deepcopy(parser)
    probe._defaults.clear()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    supplied = vars(probe.parse_args(tokens))
    forbidden = {'n_hidden', 'n_layers', 'n_heads', 'slice_num', 'mlp_ratio', 'cdpa_mode', 'front_blocks', 'front_latent_mode', 'history_mode', 'kernel_rank', 'profile', 'dropout', 'unified_pos'} & supplied.keys()
    if forbidden:
        raise ValueError('loop has independent model options: ' + str(sorted(forbidden)))
    args.eval = int(evaluation or getattr(args, 'eval', False))
    if args.eval and args.resume:
        raise ValueError('eval and resume are separate actions')
    if args.checkpoint and (not (args.eval or args.resume)):
        raise ValueError('checkpoint selection requires eval/resume')
    args.checkpoint = args.checkpoint or ('latest' if args.resume else 'final')
    args.linearno_run_dir = args.linearno_run_dir or getattr(args, 'run_dir', None)
    options = {k: v for k, v in loop_explicit.items() if k in OPTIONS}
    if 'linearno_rank' in supplied:
        options['actual_M'] = args.linearno_rank
    if options.get('topology_preset') in PRESETS and set(options) & set(TOPOLOGY_FIELDS):
        raise ValueError('preset/custom conflict')
    mapping = dict(linearno_hidden='model.hidden', linearno_heads='model.heads', linearno_variant='model.linearno_variant', linearno_ffn_ratio='model.ffn_ratio', linearno_dropout='model.dropout', linearno_ref='model.ref', linearno_unified_pos='model.unified_pos', nb_epochs='training.epochs', batch_size='training.batch_size', lr='training.lr', nmodel='training.nmodel', subsampling='training.subsampling', debug='training.debug', max_neighbors='training.max_neighbors', preprocessed='training.preprocessed', seed='runtime.seed', save_name='runtime.save_name')
    if task == 'airfrans':
        mapping['r'] = 'training.r'
    overrides = {path: supplied[k] for k, path in mapping.items() if k in supplied}
    if 'model.unified_pos' in overrides:
        overrides['model.unified_pos'] = bool(overrides['model.unified_pos'])
    if args.eval or args.resume:
        if args.linearno_run_dir is None:
            raise ValueError('loop eval/resume requires explicit experiment directory')
        root = read_metadata(args.linearno_run_dir / 'architecture.json')
        assertions = dict(options, task=task)
        if 'linearno_loop' in loop_explicit:
            assertions['linearno_loop'] = loop_explicit['linearno_loop'] == '1'
        if 'linearno_profile' in supplied:
            assertions['profile'] = args.linearno_profile
        config = restore_config(root, explicit=assertions)['config']
        for path, value in overrides.items():
            section, field = path.split('.')
            if value != config['profile_spec']['values'][section][field]:
                raise ValueError('explicit ' + path + ' conflicts with metadata')
        data = root['data_spec']['runtime']
        for field in ('task',) if task == 'airfrans' else ('fold_id', 'cfd_mesh', 'r', 'val_iter'):
            stored = data['task_variant'] if field == 'task' else data[field]
            if field in supplied and supplied[field] != stored:
                raise ValueError('explicit ' + field + ' conflicts with saved data protocol')
            setattr(args, field, stored)
        from .versioning import checkpoint_api
        checkpoint_module = checkpoint_api(config)
        if task == 'car':
            saved, path = checkpoint_module.inspect_checkpoint(args.linearno_run_dir, args.checkpoint, expected=config)
            args._linearno_metadata = saved
            args._linearno_checkpoint = path
        else:
            args._linearno_metadata = root
            if args.checkpoint != 'latest' and args.resume:
                raise ValueError('Air ensemble resume requires latest')
            from .industrial_state import inspect_members
            members = inspect_members(args.linearno_run_dir, root, evaluation=bool(args.eval), selector=args.checkpoint, checkpoint_module=checkpoint_module)
            args._linearno_checkpoint = next((pair[1] for pair in reversed(members) if pair is not None))
    else:
        options.setdefault('architecture', 'partial_share_feature_gate_v5')
        config = resolve_config(task, getattr(args, 'linearno_profile', DEFAULT_PROFILE), options=options, profile_overrides=overrides)
    base = config['profile_spec']
    tr = base['values']['training']
    m = base['values']['model']
    if tr['batch_size'] != 1:
        raise ValueError('industrial loop supports one graph per forward, batch_size=1')
    if 'weight' in supplied and args.weight != base['values']['objective']['surface_weight']:
        raise ValueError('loop does not change selected profile objective surface weight')
    if task == 'airfrans':
        loop = config['loop_spec']
        model = base['values']['model']
        training = base['values']['training']
        args.linearno_profile = base['profile']
        args.linearno_variant = loop['variant']
        args.linearno_rank = loop['actual_M']
        args.n_hidden = loop['hidden_width']
        args.n_layers = loop['unique_depth']
        args.n_heads = loop['heads']
        args.mlp_ratio = model['ffn_ratio']
        args.dropout = model['dropout']
        args.unified_pos = int(model['unified_pos'])
        args.ref = model['ref']
        args.nb_epochs = training['epochs']
        args.batch_size = training['batch_size']
        args.lr = training['lr']
        args.nmodel = training['nmodel']
        args.weight = base['values']['objective']['surface_weight']
        args.subsampling = training['subsampling']
        args.r = training['r']
        args.max_neighbors = training['max_neighbors']
        args.debug = training['debug']
        args.save_name = base['values']['runtime']['save_name']
        args.seed = base['values']['runtime']['seed']
        args.linearno_family = 'linearno'
        args.linearno_task = 'airfrans'
        if args.task not in ('full', 'scarce', 'reynolds', 'aoa'):
            raise ValueError('invalid Air task')
        args.data_path = args.my_path
    else:
        if not 0 <= args.fold_id <= 8 or tr['preprocessed'] not in (0, 1) or m['unified_pos']:
            raise ValueError('Car requires fold0..8, preprocessed0/1 and native non-unified seven channels')
        args.val_iter = getattr(args, 'val_iter', 10)
        if args.val_iter < 1:
            raise ValueError('val_iter must be positive')
        args.nb_epochs = tr['epochs']
        args.lr = tr['lr']
        args.batch_size = 1
        args.preprocessed = tr['preprocessed']
        args.seed = base['values']['runtime']['seed']
        args.weight = base['values']['objective']['surface_weight']
    args.linearno_family = 'linearno_loop'
    args.linearno_task = task
    args.linearno_profile = base['profile']
    args._linearno_loop_config = config
    args._linearno_config = base
    args._linearno_model_spec = config['model_spec']
    loop_spec = config['loop_spec']
    args.linearno_rank = loop_spec.get('actual_M', loop_spec.get('resolved_rank'))
    args.n_layers = loop_spec['unique_depth']
    identifier = run_directory_id(config)
    if args.linearno_run_dir is None:
        from cdlno.experiment import timestamp
        family_dir = 'partial_share_feature_gate_v5' if config.get('architecture') == 'partial_share_feature_gate_v5' else 'linearno_loop'
        args.linearno_run_dir = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT / 'output')) / task / family_dir / (identifier + '__' + timestamp() + '_' + uuid4().hex[:8])
    args.linearno_run_dir = Path(args.linearno_run_dir).resolve()
    if identifier not in args.linearno_run_dir.name:
        raise ValueError('loop run directory must contain ' + identifier)
    if not (args.eval or args.resume) and args.linearno_run_dir.exists():
        raise ValueError('new loop directory already exists')
    if getattr(args, 'gpu', None) is not None and args.gpu < 0:
        raise ValueError('gpu must be nonnegative')
    import torch
    import numpy as np
    args.device = args.device or (f'cuda:{(args.gpu if args.gpu is not None else 0)}' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and (not torch.cuda.is_available())):
        raise ValueError('unsupported/unavailable device')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args
