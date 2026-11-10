"""Dataset-free checks for the eight benchmark entry points and their routing."""
import argparse
import ast
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / 'PDE-Solving-StandardBenchmark'
AIR = ROOT / 'Airfoil-Design-AirfRANS'
CAR = ROOT / 'Car-Design-ShapeNetCar'
TASKS = {'airfoil': 'airfoil', 'darcy': 'darcy', 'elas': 'elasticity',
         'ns': 'ns', 'pipe': 'pipe', 'plas': 'plasticity'}


def load_router(path):
    spec = importlib.util.spec_from_file_location('_benchmark_router', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry_parser(path):
    """Execute the real argument declarations without importing a dataset/model."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    declarations = [node for node in tree.body if
                    isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'parser'
                        for target in node.targets) or
                    isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and ast.unparse(node.value.func) == 'parser.add_argument']
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['parser']


def fake_module(name, **functions):
    module = types.ModuleType(name)
    module.__dict__.update(functions)
    return module


class BenchmarkEntryTests(unittest.TestCase):
    def test_all_benchmark_sources_compile(self):
        for directory in (STANDARD, AIR, CAR):
            for path in directory.rglob('*.py'):
                with self.subTest(path=path.relative_to(ROOT)):
                    compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_standard_defaults_select_existing_transolver_models(self):
        router = load_router(STANDARD / 'cdlno_entry.py')
        for short, task in TASKS.items():
            with self.subTest(task=task):
                parser = entry_parser(STANDARD / f'exp_{short}.py')
                args = router.parse_args(parser, task, [])
                expected = ('Transolver_Irregular_Mesh' if task == 'elasticity'
                            else 'Transolver_Structured_Mesh_2D')
                self.assertEqual(args.model, expected)

    def test_standard_linearno_and_loop_forward_task_and_tokens(self):
        router = load_router(STANDARD / 'cdlno_entry.py')
        for short, task in TASKS.items():
            selection = ('LinearNO_Irregular_Mesh' if task == 'elasticity'
                         else 'LinearNO_Structured_Mesh_2D')
            for loop in (False, True):
                with self.subTest(task=task, loop=loop):
                    calls = []
                    def pure(parser, received_task, tokens):
                        calls.append((received_task, tokens))
                        return 'pure'
                    def recurrent(parser, received_task, tokens, selected_model):
                        calls.append((received_task, tokens, selected_model))
                        return 'loop'
                    modules = {
                        'cdlno.linearno.standard_entry': fake_module('pure', parse_args=pure),
                        'cdlno.linearno_loop.standard_entry': fake_module('loop', intercept=recurrent),
                    }
                    tokens = ['--model', selection] + (['--linearno-loop', '1'] if loop else [])
                    with patch.dict(sys.modules, modules):
                        result = router.parse_args(entry_parser(STANDARD / f'exp_{short}.py'), task, tokens)
                    self.assertEqual(result, 'loop' if loop else 'pure')
                    self.assertEqual(calls[0][:2], (task, tokens))
                    if loop:
                        self.assertEqual(calls[0][2], selection)

    def test_industrial_baselines_and_linearno_route_before_dataset_access(self):
        for directory, router_path, option, task in (
                (AIR, AIR / 'cdlno_entry.py', '--model', 'airfrans'),
                (CAR, CAR / 'models/cdlno_run.py', '--cfd_model', 'car')):
            router = load_router(router_path)
            for evaluation in (False, True):
                entry = directory / ('main_evaluation.py' if evaluation else 'main.py')
                with self.subTest(task=task, evaluation=evaluation):
                    args = router.parse_args(entry_parser(entry), evaluation=evaluation, argv=[])
                    self.assertEqual(getattr(args, option[2:]), 'Transolver')
                    calls = []
                    def pure(parser, tokens, *, evaluation=False):
                        calls.append((tokens, evaluation))
                        return 'pure'
                    module = f'cdlno.linearno.{"air_entry" if task == "airfrans" else "car_entry"}'
                    with patch.dict(sys.modules, {module: fake_module(module, parse_args=pure)}):
                        result = router.parse_args(entry_parser(entry), evaluation=evaluation,
                                                   argv=[option, 'LinearNO'])
                    self.assertEqual(result, 'pure')
                    self.assertEqual(calls, [([option, 'LinearNO'], evaluation)])

    def test_saved_loop_metadata_routes_eval_without_requiring_model_flag(self):
        with tempfile.TemporaryDirectory(prefix='.benchmark-', dir=ROOT / 'tests') as temporary:
            (Path(temporary) / 'architecture.json').write_text(
                json.dumps({'family': 'linearno_loop'}), encoding='utf-8')
            router = load_router(STANDARD / 'cdlno_entry.py')
            calls = []
            def recurrent(parser, task, tokens, selected_model):
                calls.append(selected_model)
                return 'loop'
            module = 'cdlno.linearno_loop.standard_entry'
            tokens = ['--experiment-dir', temporary, '--eval', '1']
            with patch.dict(sys.modules, {module: fake_module(module, intercept=recurrent)}):
                result = router.parse_args(entry_parser(STANDARD / 'exp_ns.py'), 'ns', tokens)
            self.assertEqual(result, 'loop')
            self.assertEqual(calls, [None])

    def test_industrial_loop_routes_explicit_flags_and_saved_metadata(self):
        with tempfile.TemporaryDirectory(prefix='.benchmark-', dir=ROOT / 'tests') as temporary:
            (Path(temporary) / 'architecture.json').write_text(
                json.dumps({'family': 'linearno_loop'}), encoding='utf-8')
            for directory, router_path, option, task in (
                    (AIR, AIR / 'cdlno_entry.py', '--model', 'airfrans'),
                    (CAR, CAR / 'models/cdlno_run.py', '--cfd_model', 'car')):
                router = load_router(router_path)
                for evaluation in (False, True):
                    with self.subTest(task=task, evaluation=evaluation):
                        calls = []
                        def recurrent(parser, tokens, **kwargs):
                            calls.append((tokens, kwargs))
                            return 'loop'
                        module = 'cdlno.linearno_loop.industrial_entry'
                        tokens = (['--experiment-dir', temporary] if evaluation else
                                  [option, 'LinearNO', '--linearno-loop', '1'])
                        entry = directory / ('main_evaluation.py' if evaluation else 'main.py')
                        with patch.dict(sys.modules, {module: fake_module(module, intercept=recurrent)}):
                            result = router.parse_args(entry_parser(entry), evaluation=evaluation, argv=tokens)
                        self.assertEqual(result, 'loop')
                        self.assertEqual(calls, [(tokens, dict(task=task, evaluation=evaluation,
                                                              selected_model=None if evaluation else 'LinearNO'))])

    def test_local_benchmark_imports_resolve_after_pruning(self):
        for directory in (STANDARD, AIR, CAR):
            for path in directory.rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if not isinstance(node, ast.ImportFrom) or not node.module:
                        continue
                    if node.module.split('.')[0] not in ('model', 'models', 'dataset', 'utils'):
                        continue
                    candidate = directory.joinpath(*node.module.split('.'))
                    with self.subTest(source=path.relative_to(ROOT), dependency=node.module):
                        self.assertTrue(candidate.is_dir() or candidate.with_suffix('.py').is_file())

    def test_removed_models_are_rejected(self):
        router = load_router(STANDARD / 'cdlno_entry.py')
        for removed in ('CDLNO', 'kcdno', 'lrsa_matched', 'msar_lno'):
            with self.subTest(model=removed), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    router.parse_args(entry_parser(STANDARD / 'exp_ns.py'), 'ns', ['--model', removed])

    def test_industrial_geometry_and_force_dependencies_are_present(self):
        required = [AIR / 'dataset/dataset.py', AIR / 'utils/metrics.py',
                    AIR / 'utils/metrics_NACA.py', AIR / 'utils/naca_generator.py',
                    AIR / 'utils/reorganize.py', AIR / 'LICENSE',
                    CAR / 'dataset/dataset.py', CAR / 'dataset/load_dataset.py',
                    CAR / 'utils/drag_coefficient.py']
        for path in required:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
