"""Observation-only experiment records; no model or data imports.

Entry points explicitly start/finish a session. Parsers and checkpoint loaders
remain usable without process hooks or filesystem side effects from this module.
"""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

_SESSIONS = {}


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def default_directory(task):
    root = Path(os.environ.get('CDLNO_RUNS_ROOT', Path(__file__).resolve().parents[1] / 'output'))
    return (root / task / timestamp()).resolve()


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'detach'):
        return _json(value.detach().cpu().tolist())
    if hasattr(value, 'tolist'):
        return _json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Explicit nonfinite value; never invalid JSON or silent zero.
    return value


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(_json(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


def session(args):
    value = _SESSIONS.get(id(args))
    return value if value is not None and value.args is args else None


def reserve_directory(args, path):
    """Adopt only this process's exact early reservation, never an arbitrary dir."""
    path = Path(path).resolve()
    active = session(args)
    if active is not None and not active.evaluation and active.directory == path:
        return
    path.mkdir(parents=True, exist_ok=False)


def start(args, task, *, evaluation=False, hparams=None):
    if session(args) is not None:
        raise RuntimeError('an experiment is already active for these arguments')
    active = Experiment(args, task, evaluation=evaluation, hparams=hparams)
    _SESSIONS[id(args)] = active
    return active


def finish(args):
    active = session(args)
    if active is not None:
        active.finish()


class Experiment:
    def __init__(self, args, task, *, evaluation=False, hparams=None):
        self.args, self.task, self.evaluation = args, task, evaluation
        self.family = getattr(args, 'model', getattr(args, 'cfd_model', 'Transolver'))
        self.run_key = 'run_dir'
        if getattr(args, 'linearno_family', None) in ('linearno', 'linearno_loop'):
            self.family, self.run_key = args.linearno_family, 'linearno_run_dir'
        self.resuming = self.family in ('linearno', 'linearno_loop') and getattr(args, 'resume', False)
        requested = getattr(args, self.run_key, None)
        self.directory = Path(requested).resolve() if requested is not None else default_directory(task)
        if evaluation or self.resuming:
            if requested is None or not (self.directory / 'architecture.json').is_file():
                raise ValueError('evaluation requires an existing run with architecture.json')
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
        setattr(args, self.run_key, self.directory)
        self.closed = False
        self.members = {}
        self.latest_epochs = {}
        self.eval_directory = None
        self.started = timestamp()
        self.clock = time.monotonic()
        self.result = dict(status='running', phase='startup', task=task,
                           started_at_utc=self.started, pid=os.getpid(), metrics={})
        self.config = dict(schema_version=1, task=task, model=self.family, run_directory=str(self.directory),
                           started_at_utc=self.started, resolved_arguments=_json(vars(args)),
                           hparams=_json(hparams), parameter_count_status='pending_model_construction',
                           parameters=None, architecture=None, environment=self._environment(),
                           code=self._code(), command=sys.argv, cwd=str(Path.cwd()))
        if self.family in ('linearno', 'linearno_loop'):
            self.result['seed'] = args.seed
            self.config['resolved_arguments'] = _json({k: v for k, v in vars(args).items()
                                                       if not k.startswith('_linearno')})
            self.config['linearno_profile'] = _json(args._linearno_config)
            if self.family == 'linearno_loop':
                self.config['loop_config'] = _json(args._linearno_loop_config)
        if evaluation:
            if self.family in ('linearno', 'linearno_loop'):
                self.result['resolved_arguments'] = self.config['resolved_arguments']
            else:
                self.result['resolved_arguments'] = _json(vars(args))
            self.result['environment'] = self.config['environment']
            self.result['code'] = self.config['code']
            self.result['command'] = sys.argv
            self.result['hparams'] = _json(hparams)
            self._ensure_eval()
        elif not self.resuming:
            write_json(self.directory / 'config.json', self.config)
        if self.resuming:
            self.config = json.loads((self.directory / 'config.json').read_text())
            self.result['resume_from'] = str(args._linearno_checkpoint)
        self._write_result()
        self.stdout, self.stderr, self.exception_hook = sys.stdout, sys.stderr, sys.excepthook
        log_path = (self.eval_directory / 'eval.log') if evaluation else (self.directory / 'train.log')
        self.log = log_path.open('a' if self.resuming else 'x', encoding='utf-8')
        sys.stdout, sys.stderr = _Tee(self.stdout, self.log), _Tee(self.stderr, self.log)
        sys.excepthook = self._exception
        atexit.register(self._unfinished)
        print(self.family + ' experiment:', self.directory)
        print(self.family + ' phase:', 'evaluation' if evaluation else 'training')
        if 'seed' in self.result:
            print(self.family + ' seed:', self.result['seed'])

    @staticmethod
    def _environment():
        versions = {}
        for name in ('torch', 'numpy', 'torch-geometric'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        result = dict(python=platform.python_version(), executable=sys.executable,
                      platform=platform.platform(), packages=versions,
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        torch = sys.modules.get('torch')
        if torch is not None:
            result['torch_cuda_build'] = torch.version.cuda
        return result

    @staticmethod
    def _code():
        from .provenance import code_state
        return code_state()

    def _ensure_eval(self):
        if self.eval_directory is None:
            parent = self.directory / 'evaluations'
            parent.mkdir(exist_ok=True)
            self.eval_directory = parent / timestamp()
            self.eval_directory.mkdir(exist_ok=False)
        return self.eval_directory

    @property
    def result_dir(self):
        # Training's existing figures need no evaluation record until score runs.
        return str(self._ensure_eval() if self.evaluation else self.directory / 'training_artifacts') + '/'

    def attach_model(self, model, *, hparams=None, member=0, protocol=None):
        params = dict(total=sum(p.numel() for p in model.parameters()),
                      trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        parameter = next(model.parameters())
        if self.family in ('linearno', 'linearno_loop'):
            actual = dict(parameters=params, architecture=self.args._linearno_model_spec,
                          wrapper_architecture=dict(task=self.task),
                          device=str(parameter.device), dtype=str(parameter.dtype))
        else:
            actual = dict(parameters=params,
                          architecture=dict(class_path=type(model).__module__ + '.' + type(model).__qualname__),
                          wrapper_architecture=dict(task=self.task),
                          device=str(parameter.device), dtype=str(parameter.dtype))
        self.members[str(member)] = actual
        if not self.evaluation and not self.resuming:
            self.config.update(parameter_count_status='measured', parameters=params,
                               architecture=actual['architecture'], wrapper_architecture=actual['wrapper_architecture'],
                               members=self.members)
            if hparams is not None:
                self.config['hparams'] = _json(hparams)
            if protocol is not None:
                self.config.setdefault('protocol', {}).update(_json(protocol))
            self.config['constructed_members_parameters'] = {
                key: sum(item['parameters'][key] for item in self.members.values())
                for key in ('total', 'trainable')}
            write_json(self.directory / 'config.json', self.config)
        else:
            self.result['loaded_members'] = self.members
        self.result['phase'] = 'model_ready'
        self._write_result()

    def update_protocol(self, values):
        if not self.evaluation and not self.resuming:
            self.config.setdefault('protocol', {}).update(_json(values))
            write_json(self.directory / 'config.json', self.config)

    def record_training_setup(self, optimizer, scheduler):
        if self.evaluation or self.resuming:
            return
        self.config['optimizer'] = dict(
            type=type(optimizer).__module__ + '.' + type(optimizer).__name__,
            parameter_groups=[{k: _json(v) for k, v in group.items() if k != 'params'}
                              for group in optimizer.param_groups])
        self.config['scheduler'] = dict(type=type(scheduler).__module__ + '.' + type(scheduler).__name__)
        write_json(self.directory / 'config.json', self.config)

    def record_epoch(self, epoch, metrics, *, member=0):
        if self.evaluation or self.closed:
            raise RuntimeError('epoch records require an active training run')
        if not self.members:
            raise RuntimeError('attach the real model before recording training')
        row = dict(epoch=int(epoch), member=member, metrics=_json(metrics), recorded_at_utc=timestamp())
        with (self.directory / 'train_history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.latest_epochs[str(member)] = row
        self.result.update(phase='training', latest_epochs=self.latest_epochs)
        self._write_result()

    def record_metrics(self, metrics):
        self.result['metrics'].update(_json(metrics))
        self._write_result()

    def visualize(self, model, completed_epoch, total_epochs, *, member=0, **task_inputs):
        """Optional diagnostic side effect; never part of the training objective."""
        if self.evaluation:
            return
        from .periodic_visualization import PeriodicFields, due
        if not due(completed_epoch, total_epochs):
            return
        if not hasattr(self, '_field_visualizers'):
            self._field_visualizers = {}
        if member not in self._field_visualizers:
            name = {'linearno': 'LinearNO', 'linearno_loop': 'PhyLoopNO'}.get(self.family, 'Transolver')
            self._field_visualizers[member] = PeriodicFields(self.directory, self.task, name,
                                                           seed=getattr(self.args, 'seed', None), member=member)
        event = self._field_visualizers[member].after_epoch(model, completed_epoch, total_epochs, **task_inputs)
        self.result.setdefault('visualization_events', []).append(event)
        self._write_result()

    def record_air_scores(self, result_dir):
        """Import the evaluator's exact structured output, never rederive metrics."""
        path = Path(result_dir) / 'score.json'
        scores = json.loads(path.read_text())
        self.record_metrics(dict(score=scores, score_file=str(path.relative_to(self.directory))))

    def _write_result(self):
        if self.evaluation:
            write_json(self._ensure_eval() / 'results.json', self.result)
            # Serialize concurrent evaluation index updates. Individual attempts
            # are authoritative, including failed/interrupted ones.
            from .runtime import exclusive_lock
            with exclusive_lock(self.directory / '.eval_index.lock', blocking=True):
                evaluations = []
                for path in sorted((self.directory / 'evaluations').glob('*/results.json')):
                    evaluations.append(dict(result_file=str(path.relative_to(self.directory)),
                                            **json.loads(path.read_text())))
                write_json(self.directory / 'eval_results.json', dict(evaluations=evaluations))
        else:
            write_json(self.directory / 'train_results.json', self.result)

    def finish(self, *, status='completed', error=None):
        if self.closed:
            return
        self.result.update(status=status, finished_at_utc=timestamp(),
                           elapsed_wall_seconds=time.monotonic() - self.clock)
        if not self.evaluation:
            self.result['final_recorded_metrics_by_member'] = {
                key: row['metrics'] for key, row in self.latest_epochs.items()}
        if error is not None:
            self.result['error'] = error
        self._write_result()
        self.closed = True
        atexit.unregister(self._unfinished)
        sys.stdout, sys.stderr, sys.excepthook = self.stdout, self.stderr, self.exception_hook
        self.log.close()
        _SESSIONS.pop(id(self.args), None)

    def _exception(self, kind, value, tb):
        try:
            self.exception_hook(kind, value, tb)
        finally:
            self.finish(status='interrupted' if issubclass(kind, KeyboardInterrupt) else 'failed',
                        error=f'{kind.__name__}: {value}')

    def _unfinished(self):
        if not self.closed:
            self.finish(status='incomplete', error='process exited before explicit completion')
