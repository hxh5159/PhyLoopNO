"""Public launch commands preserve paper settings and explicit task paths."""
from pathlib import Path
from unittest.mock import patch
import pytest
from phyloopno.cli import parse, plan, execute, TASKS


@pytest.mark.parametrize('model', ['phyloopno', 'linearno', 'transolver'])
@pytest.mark.parametrize('task', TASKS)
def test_all_task_model_plans(task, model):
    tokens = [task, 'train', '--model', model]
    if model == 'phyloopno' and task in ('car', 'airfrans'):
        tokens += ['--experts', '3']
    _, args, extra = parse(tokens)
    value = plan(args, extra)
    assert value['task'] == task and value['model'] == model
    if model == 'phyloopno':
        assert value['config']['loop_spec']['executed_depth'] == 8
        assert value['config']['prefix_blocks'] == value['config']['suffix_blocks'] == 1


@pytest.mark.parametrize('task,expected', [('pipe', 768), ('ns', 1024)])
def test_wide_ffn_uses_selected_profile(task, expected):
    _, args, extra = parse([task, 'train', '--ablation', 'wide_ffn', '--profile', 'transolver_matched'])
    cfg = plan(args, extra)['config']
    assert cfg['expert_count'] == 1 and cfg['expert_width'] == expected


@pytest.mark.parametrize('flag', ['--depth', '--expert-width', '--experts'])
def test_zero_dimensions_rejected(flag):
    with pytest.raises(SystemExit):
        parse(['airfoil', 'train', '--ablation', 'wide_ffn', flag, '0'])


def test_paths_are_resolved_before_changing_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, args, extra = parse(['airfoil', 'train', '--data-path', 'relative data'])
    value = plan(args, extra)
    assert str(tmp_path / 'relative data') in value['argv']
    assert len(Path(value['run']).name) < 100


@pytest.mark.parametrize('ablation,qk,norm,e', [
    ('main', 'independent', 'visit_independent', 4),
    ('full_share', 'shared', 'shared', 1),
    ('selective_share', 'independent', 'visit_independent', 1),
    ('shared_slicing', 'shared', 'visit_independent', 4)])
def test_sharing_selectors(ablation, qk, norm, e):
    _, args, extra = parse(['airfoil', 'train', '--ablation', ablation])
    cfg = plan(args, extra)['config']
    assert (cfg['qk_sharing'], cfg['core_norm_mode'], cfg['expert_count']) == (qk, norm, e)


def test_industrial_transolver_gpu_selection(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    value = dict(task='car', model='transolver', gpu=1, entry='main.py', argv=[], cwd='.')
    with patch('phyloopno.cli.subprocess.run') as run:
        run.return_value.returncode = 0
        assert execute(value) == 0
    assert run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == '1'
