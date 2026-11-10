import argparse, yaml, json
import torch
import train
import utils.metrics as metrics
from dataset.dataset import Dataset
import os.path as osp
import numpy as np
from cdlno_entry import parse_args as parse_task_args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=('Transolver', 'LinearNO'), default='Transolver', help='Transolver baseline or LinearNO / PhyLoopNO with loop flags')
parser.add_argument('-n', '--nmodel', help='Number of trained models for standard deviation estimation (default: 1)', default=1, type=int)
parser.add_argument('-w', '--weight', help='Weight in front of the surface loss (default: 1)', default=1, type=float)
parser.add_argument('-t', '--task', help='Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default='full', type=str)
parser.add_argument('-s', '--score', help='If you want to compute the score of the models on the associated test set. (default: 0)', default=0, type=int)
parser.add_argument('--my_path', default='/data/path', type=str)
parser.add_argument('--save_path', default='metrics', type=str)
args = parse_task_args(parser)
if args.model == 'LinearNO':
    from cdlno.linearno.air_entry import run_cli
    run_cli(args)
    raise SystemExit(0)
with open(args.my_path + '/manifest.json', 'r') as f:
    manifest = json.load(f)
manifest_train = manifest[args.task + '_train']
test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
n = int(0.1 * len(manifest_train))
train_dataset = manifest_train[:-n]
val_dataset = manifest_train[-n:]
print('start load data')
train_dataset, coef_norm = Dataset(train_dataset, norm=True, sample=None, my_path=args.my_path)
val_dataset = Dataset(val_dataset, sample=None, coef_norm=coef_norm, my_path=args.my_path)
print('load data finish')
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')
with open('params.yaml', 'r') as f:
    hparams = yaml.safe_load(f)[args.model]
models = []
for i in range(args.nmodel):
    if args.model == 'Transolver':
        from models.Transolver import Transolver
        model = Transolver(n_hidden=256, n_layers=8, space_dim=7, fun_dim=0, n_head=8, mlp_ratio=2, out_dim=4, slice_num=32, unified_pos=1).cuda()
    log_path = osp.join(args.save_path, args.task, args.model)
    print('start training')
    model = train.main(device, train_dataset, val_dataset, model, hparams, log_path, criterion='MSE_weighted', val_iter=10, reg=args.weight, name_mod=args.model, val_sample=True)
    print('end training')
    models.append(model)
torch.save(models, osp.join(args.save_path, args.task, args.model, args.model))
score_path = 'scores'
score_array_path = osp.join('scores', args.task)
if bool(args.score):
    print('start score')
    s = args.task + '_test' if args.task != 'scarce' else 'full_test'
    coefs = metrics.Results_test(device, [models], [hparams], coef_norm, args.my_path, path_out=score_path, n_test=3, criterion='MSE', s=s)
    np.save(osp.join(score_array_path, 'true_coefs'), coefs[0])
    np.save(osp.join(score_array_path, 'pred_coefs_mean'), coefs[1])
    np.save(osp.join(score_array_path, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join(score_array_path, 'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join(score_array_path, 'surf_coefs_' + str(n)), file)
    np.save(osp.join(score_array_path, 'true_bls'), coefs[5])
    np.save(osp.join(score_array_path, 'bls'), coefs[6])
    print('end score')
