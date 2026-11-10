"""Print or sequentially run the manuscript's three-model depth matrix."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=('airfoil', 'darcy', 'elasticity', 'plasticity'))
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=ROOT / 'output/depth')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    for model in ('transolver', 'linearno', 'phyloopno'):
        for depth in (8, 16, 24, 32, 40):
            common = [sys.executable, str(ROOT / 'scripts/run.py'), args.task,
                      '--model', model, '--depth', str(depth), '--data-root', str(args.data_root.resolve()),
                      '--gpu', str(args.gpu)]
            if model == 'transolver':
                name = f'{args.task}_transolver_depth{depth}_seed0'
                for action in ('train', 'eval'):
                    command = [*common, action, '--save_name', name]
                    print(shlex.join(command), flush=True)
                    if args.execute:
                        subprocess.run(command, check=True)
            else:
                run = args.output_root.resolve() / args.task / model / f'L{depth}_seed0'
                command = [*common, 'train_eval', '--seed', '0', '--experiment-dir', str(run)]
                print(shlex.join(command), flush=True)
                if args.execute:
                    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
