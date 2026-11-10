"""Fingerprint the published checkout without requiring its development history."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = (
    'cdlno', 'linearno_loop', 'tran_evaluate', 'phyloopno', 'iclr_vis',
    'PDE-Solving-StandardBenchmark', 'Airfoil-Design-AirfRANS',
    'Car-Design-ShapeNetCar',
)


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def code_state():
    """Git identity is optional for source archives; never invent a commit."""
    try:
        top = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip()
        if Path(top).resolve() != ROOT:
            return dict(commit=None, dirty=None)
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=5))
        return dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return dict(commit=None, dirty=None)


def source_manifest():
    """Hash actual retained code, configuration and launchers in stable order."""
    paths = set()
    for name in SOURCE_DIRECTORIES:
        directory = ROOT / name
        if directory.is_dir():
            paths.update(path for path in directory.rglob('*')
                         if path.is_file() and path.suffix in ('.py', '.sh', '.yaml', '.yml', '.toml')
                         and '__pycache__' not in path.parts)
    for name in ('pyproject.toml', 'requirements.txt', 'requirements-industrial.txt'):
        if (ROOT / name).is_file():
            paths.add(ROOT / name)
    paths.update(ROOT.glob('*.py'))
    paths.update(ROOT.glob('*.sh'))
    raw, normalized = {}, {}
    for path in sorted(paths):
        name = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        raw[name] = hashlib.sha256(data).hexdigest()
        if path.suffix == '.py':
            text = ast.dump(ast.parse(data.decode('utf-8-sig')), include_attributes=False)
            normalized[name] = hashlib.sha256(text.encode()).hexdigest()
        else:
            normalized[name] = raw[name]
    return raw, normalized


def provenance(task=None):
    """Use the stable metadata interface with fingerprints of this release.

    ``normalized_patch_sha256`` is retained as a wire field name. Its value now
    fingerprints the current normalized source manifest, with no git diff,
    historical commit lookup or removed-family source projection.
    """
    from .experiment import Experiment
    code = code_state()
    sources, normalized = source_manifest()
    return dict(
        target_sha=code['commit'], base_commit=code['commit'], dirty=code['dirty'],
        transolver_sha='75e0f67643806a81cd1d3f6adc88dd8c02416fe7',
        linearno_sha='3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269',
        paper_version='2511.06294v3',
        paper_sha256='637e2953c0e223df7ffe2be92dd3825fccf55232cd73e2c955935a3944a470fd',
        source_sha256=_digest(sources), normalized_patch_sha256=_digest(normalized),
        command=list(sys.argv), environment=Experiment._environment())
