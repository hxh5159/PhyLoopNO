"""Route ShapeNet-Car runs without loading data or allocating a model."""
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


def parse_args(parser, *, evaluation=False, argv=None):
    tokens = sys.argv[1:] if argv is None else list(argv)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--cfd_model')
    selected, _ = selector.parse_known_args(tokens)
    if _loop_intent(tokens, evaluation):
        from cdlno.linearno_loop.industrial_entry import intercept
        result = intercept(parser, tokens, task='car', evaluation=evaluation, selected_model=selected.cfd_model)
        if result is not None:
            return result
    if selected.cfd_model == 'LinearNO':
        from cdlno.linearno.car_entry import parse_args as parse_linearno
        return parse_linearno(parser, tokens, evaluation=evaluation)
    args = parser.parse_args(tokens)
    if args.cfd_model != 'Transolver':
        parser.error('choose Transolver or LinearNO; PhyLoopNO uses --linearno-loop 1')
    return args
