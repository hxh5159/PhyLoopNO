"""Select the original Transolver, LinearNO, or PhyLoopNO task adapter."""
import argparse
import json
from pathlib import Path
import sys


def _loop_intent(tokens, evaluation=False):
    if any(token.split('=')[0].startswith('--linearno-loop') for token in tokens):
        return True
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument('--experiment-dir', '--linearno-run-dir', '--run_dir', dest='directory', type=Path)
    probe.add_argument('--eval', type=int, default=0)
    probe.add_argument('--resume', action='store_true')
    intent, _ = probe.parse_known_args(tokens)
    if (evaluation or intent.eval or intent.resume) and intent.directory is not None:
        sidecar = intent.directory / 'architecture.json'
        if sidecar.is_file():
            return json.loads(sidecar.read_text(encoding='utf-8')).get('family') == 'linearno_loop'
    return False


def parse_args(parser, task, argv=None):
    tokens = sys.argv[1:] if argv is None else list(argv)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--model')
    selected, _ = selector.parse_known_args(tokens)
    if _loop_intent(tokens):
        from cdlno.linearno_loop.standard_entry import intercept
        result = intercept(parser, task, tokens, selected.model)
        if result is not None:
            return result
    if selected.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        from cdlno.linearno.standard_entry import parse_args as parse_linearno
        return parse_linearno(parser, task, tokens)
    args = parser.parse_args(tokens)
    allowed = ('Transolver_Irregular_Mesh', 'Transolver_Structured_Mesh_2D')
    if args.model not in allowed:
        parser.error('choose a Transolver model or a LinearNO model; PhyLoopNO uses --linearno-loop 1')
    return args
