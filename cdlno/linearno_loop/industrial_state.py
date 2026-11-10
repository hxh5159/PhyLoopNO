"""Industrial PhyLoopNO metadata, independent RNGs and strict member archives."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import torch
from cdlno.linearno.schema import unpack_state
from cdlno.linearno.checkpoint import restore_random_state, strict_load, sha256
from cdlno.training_state import _optimizer_signature, _atomic
from linearno_loop.contracts import digest, seal, require_equal
from linearno_loop.versioning import make_metadata, read_metadata, write_metadata, validate_constructor
from .checkpoint import inspect_checkpoint, read_pair, validate_optimizer_state

def generators(args):
    seeds = args._linearno_loop_config['fair_comparison']['dataloader_generators']
    return {k: torch.Generator().manual_seed(seeds[k]) for k in ('train', 'test')}

def member_seed(args, member):
    if type(member) is not int or member < 0:
        raise ValueError('invalid member index')
    return args.seed if member == 0 else int(digest(dict(seed=args.seed, task=args.linearno_task, member=member))[:15], 16)

def construct(args, member=0):
    from .versioning import construct as versioned_construct
    return versioned_construct(args._linearno_loop_config, member_seed=member_seed(args, member))

def data_contract(config, native):
    value = copy.deepcopy(native)
    return dict(protocol=config['profile_spec']['values']['data'], split=value.pop('split'), sampling=value.pop('sampling'), checksums=value.pop('checksums'), scope='synthetic' if 'SYNTHETIC' in native['split'].upper() else 'real', runtime=value)

def legacy_data_view(args):
    """Read-only view solely for the unchanged dataset checksum/normalizer helper."""
    other = copy.copy(args)
    if hasattr(args, '_linearno_metadata'):
        saved = copy.deepcopy(args._linearno_metadata)
        data = saved['data_spec']
        saved['data_spec'] = {**data['runtime'], **{k: data[k] for k in ('split', 'sampling', 'checksums')}}
        other._linearno_metadata = saved
    return other

def provenance(task):
    from cdlno.provenance import provenance as current
    return current(task)

def restore_training(saved, path, model, optimizer, scheduler, *, steps, generators, current_provenance):
    directory = Path(path).parent
    newest = max((int(p.stem.split('_')[1]) for p in directory.glob('epoch_*.json') if '.metadata.' not in p.name))
    state = saved['resume_state']
    epoch = state['epoch']
    if epoch != newest:
        raise ValueError('resume requires newest committed epoch')
    if saved['provenance_spec']['source_sha256'] != current_provenance['source_sha256']:
        raise ValueError('resume source code differs')
    expected = saved['data_spec']['runtime']['optimizer_signature']
    require_equal(expected, json.loads(json.dumps(_optimizer_signature(model, optimizer))), 'optimizer_signature')
    opt = unpack_state(state['optimizer'])
    sch = unpack_state(state['scheduler'])
    from .versioning import checkpoint_api
    checkpoint = checkpoint_api(saved['resolved_config'])
    checkpoint.validate_optimizer_state(opt, optimizer, model=model)
    if state['global_step'] != epoch * steps or sch['last_epoch'] != epoch * steps or sch['_step_count'] != epoch * steps + 1:
        raise ValueError('optimizer/scheduler progress mismatch')
    if sch['total_steps'] != scheduler.total_steps:
        raise ValueError('scheduler construction mismatch')
    _, weights = checkpoint.read_pair(path, expected=saved['resolved_config'])
    strict_load(model, weights)
    optimizer.load_state_dict(opt)
    scheduler.load_state_dict(sch)
    restore_random_state(state, generators)
    return epoch

def inspect_members(directory, root, *, evaluation, selector, checkpoint_module=None):
    if checkpoint_module is None:
        from .versioning import checkpoint_api
        checkpoint_module = checkpoint_api(root['resolved_config'])
    directory = Path(directory)
    count = root['profile_spec']['values']['training']['nmodel']
    result = []
    unfinished = False
    expected_names = {f'member_{i:03d}' for i in range(count)}
    if any((p.name not in expected_names for p in directory.glob('member_*'))):
        raise ValueError('unexpected ensemble member directory')
    for i in range(count):
        member = directory / f'member_{i:03d}'
        pointer = member / 'checkpoints/latest.json'
        if not pointer.exists():
            if evaluation or i == 0:
                raise ValueError('ensemble has no committed checkpoint for member ' + str(i))
            if list((member / 'checkpoints').glob('*.pt')) or list((member / 'weights').glob('*.pt')):
                raise ValueError('orphaned member state files require inspection')
            unfinished = True
            result.append(None)
            continue
        if unfinished:
            raise ValueError('ensemble progress is not sequential')
        metadata, path = checkpoint_module.inspect_checkpoint(member, selector, expected=root['resolved_config'])
        for key in ('model_spec', 'profile_spec', 'loop_spec', 'data_spec', 'normalizer_spec', 'provenance_spec'):
            require_equal(root[key], metadata[key], 'member.' + key)
        state = metadata['resume_state']
        sampler = unpack_state(state['sampler_state'])
        if sampler.get('member') != i:
            raise ValueError('checkpoint ensemble member identity mismatch')
        if state['checkpoint_role'] != 'final':
            unfinished = True
        result.append((metadata, path))
    if evaluation:
        ensemble = json.loads((directory / 'ensemble.json').read_text())
        if set(ensemble) != {'family', 'config_hash', 'checkpoint_role', 'members'} or ensemble['family'] != 'linearno_loop' or ensemble['config_hash'] != root['config_hash'] or (ensemble['checkpoint_role'] != 'final'):
            raise ValueError('ensemble manifest family/config mismatch')
        rows = ensemble['members']
        if len(rows) != count:
            raise ValueError('ensemble member count mismatch')
        for i, (row, pair) in enumerate(zip(rows, result)):
            _, path = pair
            manifest = json.loads(path.read_text())
            weight = path.parent.parent / manifest['weights']['path']
            expected = dict(member_id=f'member_{i:03d}', order=i, path=str(weight.relative_to(directory)), sha256=sha256(weight), format='state_dict')
            require_equal(expected, row, 'ensemble.member')
    return result

def export_state(model, path):
    _atomic(Path(path), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
