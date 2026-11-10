"""Train/evaluate the paper model and retained baselines using native tasks."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity", "airfrans", "car")
ENTRIES = dict(airfoil="exp_airfoil.py", darcy="exp_darcy.py", elasticity="exp_elas.py",
               pipe="exp_pipe.py", ns="exp_ns.py", plasticity="exp_plas.py")
RELATIVE_DATA = dict(airfoil="fno/airfoil/naca", darcy="fno", elasticity="fno",
                     pipe="fno/pipe", ns="fno", plasticity="fno/plas_N987_T20.mat")
PAPER_EXPERTS = dict(airfoil=4, pipe=3, plasticity=4, darcy=1, ns=4, elasticity=4)


def project(task):
    return ROOT / {"car": "Car-Design-ShapeNetCar", "airfrans": "Airfoil-Design-AirfRANS"}.get(task, "PDE-Solving-StandardBenchmark")


def native_parse(task, action, entry, tokens):
    """Run only native argument declarations and the small routing module."""
    folder = project(task)
    nodes = []
    for node in ast.parse((folder / entry).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "parser":
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "parser.add_argument":
            nodes.append(node)
    scope = {"argparse": argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<native argument declarations>", "exec"), scope)
    route = folder / ("models/cdlno_run.py" if task == "car" else "cdlno_entry.py")
    spec = importlib.util.spec_from_file_location("_phyloopno_route_" + task, route)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if task in ENTRIES:
        return module.parse_args(scope["parser"], task, tokens)
    return module.parse_args(scope["parser"], evaluation=action == "eval", argv=tokens)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("task", choices=TASKS)
    p.add_argument("action", choices=("train", "resume", "eval", "train_eval"))
    p.add_argument("--model", choices=("phyloopno", "linearno", "transolver"), default="phyloopno")
    p.add_argument("--depth", type=int)
    p.add_argument("--experts", type=int)
    p.add_argument("--expert-width", type=int)
    p.add_argument("--ablation", choices=("main", "full_share", "selective_share", "shared_slicing", "wide_ffn"), default="main")
    p.add_argument("--profile", choices=("paper_table8_on_release_model", "official_release", "transolver_matched"))
    p.add_argument("--seed", type=int)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("PHYLOOPNO_DATA_ROOT", ROOT / "data")))
    p.add_argument("--data-path", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--experiment-dir", type=Path)
    p.add_argument("--checkpoint")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args(argv)
    if args.depth is not None and args.depth <= 0:
        p.error('--depth must be positive')
    if any(value is not None and value <= 0 for value in (args.experts, args.expert_width)):
        p.error('--experts and --expert-width must be positive')
    for name in ('data_root', 'data_path', 'data_dir', 'cache_dir', 'experiment_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    path_flags = {'--data_path', '--my_path', '--data_dir', '--save_dir'}
    for index, token in enumerate(extra):
        flag, separator, value = token.partition('=')
        if flag in path_flags:
            if separator:
                extra[index] = flag + '=' + str(Path(value).expanduser().resolve())
            elif index + 1 < len(extra):
                extra[index + 1] = str(Path(extra[index + 1]).expanduser().resolve())
    if args.gpu < 0:
        p.error("--gpu must be nonnegative")
    if args.model != "phyloopno" and (args.ablation != "main" or args.experts is not None or args.expert_width is not None):
        p.error("expert and ablation options require --model phyloopno")
    if args.model == "transolver" and args.action in ("resume", "train_eval"):
        p.error("native Transolver supports train/eval; use its explicit --save_name checkpoint convention")
    if args.action in ("eval", "resume") and args.model != "transolver" and args.experiment_dir is None:
        p.error("eval/resume requires --experiment-dir for one exact saved run")
    if args.model == "transolver" and (args.experiment_dir or args.checkpoint):
        p.error("native Transolver uses --save_name (Standard) or its original --model_path/--cfd_model paths")
    if args.model == "transolver" and args.seed not in (None, 0):
        p.error("the retained native Transolver entries use their original seed 0")
    forbidden = {"--model", "--cfd_model", "--eval", "--resume", "--linearno-loop", "--linearno-loop-architecture"}
    if any(token.split("=")[0] in forbidden for token in extra):
        p.error("model/action are selected by this launcher")
    return p, args, extra


def plan(args, extra):
    action = "train" if args.action == "train_eval" else args.action
    standard = args.task in ENTRIES
    entry = ENTRIES.get(args.task) or ("main_evaluation.py" if action == "eval" else "main.py")
    selector = ("Transolver_Irregular_Mesh" if args.task == "elasticity" else "Transolver_Structured_Mesh_2D") if args.model == "transolver" else ("LinearNO_Irregular_Mesh" if args.task == "elasticity" else "LinearNO_Structured_Mesh_2D")
    tokens = (["--model", selector, "--eval", str(int(action == "eval")), "--gpu", str(args.gpu)] if standard else
              ["--model" if args.task == "airfrans" else "--cfd_model", "Transolver" if args.model == "transolver" else "LinearNO"])
    if not standard and args.model != "transolver":
        tokens += ["--gpu", str(args.gpu)]
    if args.seed is not None and args.model != "transolver":
        tokens += ["--seed", str(args.seed)]
    if args.task in RELATIVE_DATA:
        tokens += ["--data_path", str(args.data_path or args.data_root / RELATIVE_DATA[args.task])]
    elif args.task == "car":
        tokens += ["--data_dir", str(args.data_dir or args.data_root / "mlcfd_data/training_data"),
                   "--save_dir", str(args.cache_dir or args.data_root / "mlcfd_data/preprocessed_data")]
    else:
        tokens += ["--my_path", str(args.data_path or args.data_root / "AirfRANS/Dataset")]
    if args.model != "transolver":
        if args.profile:
            tokens += ["--linearno-profile", args.profile]
        if args.experiment_dir:
            tokens += ["--experiment-dir", str(args.experiment_dir.resolve())]
        if args.checkpoint:
            tokens += ["--checkpoint", args.checkpoint]
        if action == "resume":
            tokens += ["--resume"]
    if args.model == "phyloopno":
        tokens += ["--linearno-loop", "1", "--linearno-loop-architecture", "partial_share_feature_gate_v5"]
        # Saved metadata is authoritative for eval/resume; only explicit options assert it.
        if action == "train" or args.depth is not None:
            tokens += ["--linearno-loop-executed-depth", str(args.depth or 8)]
        experts, width = args.experts, args.expert_width
        if action == "train" and args.ablation in ("full_share", "selective_share"):
            experts = 1
        if action == "train" and args.ablation == "wide_ffn":
            count = experts if experts is not None else PAPER_EXPERTS.get(args.task)
            if count is None:
                raise ValueError("industrial wide_ffn requires the original --experts count")
            from cdlno.linearno.profiles import resolve_config
            probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
            probe.add_argument('--n-hidden', '--linearno-hidden', dest='hidden', type=int)
            probe.add_argument('--mlp_ratio', '--linearno-ffn-ratio', dest='ratio', type=int)
            dimensions, _ = probe.parse_known_args(extra)
            base = resolve_config(args.task, args.profile or 'paper_table8_on_release_model')['values']['model']
            base_width = (dimensions.hidden or base['hidden']) * (dimensions.ratio or base['ffn_ratio'])
            experts, width = 1, count * (width or base_width)
        if experts is not None:
            tokens += ["--linearno-loop-dense-expert-count", str(experts)]
        if width is not None:
            tokens += ["--linearno-loop-dense-expert-width", str(width)]
        if action == "train" or args.ablation != "main":
            tokens += ["--linearno-loop-qk-sharing", "shared" if args.ablation in ("full_share", "shared_slicing") else "independent",
                       "--linearno-loop-core-norm-mode", "shared" if args.ablation == "full_share" else "visit_independent"]
    elif args.depth is not None:
        if not standard:
            raise ValueError("baseline depth scaling is supported for Standard benchmarks")
        tokens += ["--n-layers", str(args.depth)]
    tokens += extra
    native = native_parse(args.task, action, entry, tokens)
    directory = getattr(native, "linearno_run_dir", None)
    if directory and not args.experiment_dir:
        tokens += ["--experiment-dir", str(directory)]
    config = getattr(native, "_linearno_loop_config", getattr(native, "_linearno_config", None))
    return dict(task=args.task, action=action, model=args.model, gpu=args.gpu,
                cwd=str(project(args.task)), entry=entry,
                argv=tokens, run=str(directory) if directory else None, config=config)


def execute(value):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if value['model'] == 'transolver' and value['task'] not in ENTRIES:
        visible = env.get('CUDA_VISIBLE_DEVICES')
        if visible:
            devices = visible.split(',')
            if value['gpu'] >= len(devices):
                raise ValueError('--gpu is outside CUDA_VISIBLE_DEVICES')
            env['CUDA_VISIBLE_DEVICES'] = devices[value['gpu']]
        elif visible is None:
            env['CUDA_VISIBLE_DEVICES'] = str(value['gpu'])
    command = [sys.executable, "-u", value["entry"], *value["argv"]]
    print("Working directory:", value["cwd"], flush=True)
    print("Command:", shlex.join(command), flush=True)
    return subprocess.run(command, cwd=value["cwd"], env=env).returncode


def main(argv=None):
    parser, args, extra = parse(argv)
    try:
        value = plan(args, extra)
        if args.dry_run:
            print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
            return 0
        code = execute(value)
        if code or args.action != "train_eval":
            return code
        args.action, args.experiment_dir, args.depth = "eval", Path(value["run"]), None
        args.experts, args.expert_width, args.ablation = None, None, "main"
        return execute(plan(args, extra))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
