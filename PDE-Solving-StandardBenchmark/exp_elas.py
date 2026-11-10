import os
import argparse
from cdlno_entry import parse_args as parse_task_args
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer
parser = argparse.ArgumentParser('Training Transformer')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--model', type=str, default='Transolver_Irregular_Mesh')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--gpu', type=str, default='1', help='GPU index to use')
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='elas_Transolver')
parser.add_argument('--data_path', type=str, default='/data/fno')
args = parse_task_args(parser, 'elasticity')
eval = args.eval
save_name = args.save_name
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
    from linearno_entry import model_kwargs as linearno_model_kwargs, StandardRun as LinearNORun
    from linearno_entry import start as start_linearno, finish as finish_linearno
    from linearno_entry import normalizer as linearno_normalizer, verify_data as verify_linearno_data
    start_linearno(args, 'elasticity')

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f'Total Trainable Params: {total_params}')
    return total_params

def main():
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        verify_linearno_data(args)
    ntrain = args.ntrain
    ntest = 200
    PATH_Sigma = args.data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
    PATH_XY = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'
    input_s = np.load(PATH_Sigma)
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
    input_xy = np.load(PATH_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]
    print(input_s.shape, input_xy.shape)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        y_normalizer = linearno_normalizer(args, 'output', train_s, UnitTransformer)
    else:
        y_normalizer = UnitTransformer(train_s)
    train_s = y_normalizer.encode(train_s)
    y_normalizer.cuda()
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_xy, train_xy, train_s), batch_size=args.batch_size, shuffle=True)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_xy, test_xy, test_s), batch_size=args._linearno_config['values']['training']['test_batch_size'], shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_xy, test_xy, test_s), batch_size=args.batch_size, shuffle=False)
    print('Dataloading is over.')
    model_run = None
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        model = get_model(args).Model(**linearno_model_kwargs(args)).cuda()
        model_run = LinearNORun(args, model)
        model_run.recorder.update_protocol(dict(ntrain=ntrain, ntest=ntest))
    else:
        model = get_model(args).Model(space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden, dropout=args.dropout, n_head=args.n_heads, Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=0, out_dim=1, slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos).cuda()
    results_dir = model_run.result_dir if model_run is not None else './results/' + save_name + '/'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if model_run is not None and model_run.recorder is not None:
        model_run.recorder.record_training_setup(optimizer, scheduler)
    myloss = TestLoss(size_average=False)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        model_run.prepare(optimizer, scheduler, train_loader, test_loader)
        myloss = TestLoss(size_average=True)
    if eval:
        if model_run is not None:
            model_run.load(model)
        else:
            model.load_state_dict(torch.load('./checkpoints/' + save_name + '.pt'))
        model.eval()
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        rel_err = 0.0
        showcase = 10
        id = 0
        with torch.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                x, fx, y = (pos.cuda(), fx.cuda(), y.cuda())
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()
                if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                    rel_err += tl * y.shape[0]
                else:
                    rel_err += tl
                if id < showcase:
                    print(id)
                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[0, :, 1].detach().cpu().numpy(), c=y[0, :].detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1000)
                    plt.savefig(os.path.join(results_dir, 'gt_' + str(id) + '.pdf'), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[0, :, 1].detach().cpu().numpy(), c=out[0, :].detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1000)
                    plt.savefig(os.path.join(results_dir, 'pred_' + str(id) + '.pdf'), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[0, :, 1].detach().cpu().numpy(), c=(y[0, :] - out[0, :]).detach().cpu().numpy(), cmap='coolwarm')
                    plt.clim(-8, 8)
                    plt.colorbar()
                    plt.savefig(os.path.join(results_dir, 'error_' + str(id) + '.pdf'), bbox_inches='tight', pad_inches=0)
                    plt.close()
        rel_err /= ntest
        print('rel_err : {}'.format(rel_err))
        if model_run is not None and model_run.recorder is not None:
            model_run.recorder.record_metrics(dict(relative_l2=rel_err))
    else:
        for ep in range(args.epochs):
            if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                if ep < model_run.start_epoch:
                    continue
            model.train()
            train_loss = 0
            for pos, fx, y in train_loader:
                x, fx, y = (pos.cuda(), fx.cuda(), y.cuda())
                optimizer.zero_grad()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                loss = myloss(out, y)
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                    train_loss += loss.item() * y.shape[0]
                else:
                    train_loss += loss.item()
            scheduler.step()
            train_loss = train_loss / ntrain
            print('Epoch {} Train loss : {:.5f}'.format(ep, train_loss))
            model.eval()
            rel_err = 0.0
            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = (pos.cuda(), fx.cuda(), y.cuda())
                    out = model(x, None).squeeze(-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                        rel_err += tl * y.shape[0]
                    else:
                        rel_err += tl
            rel_err /= ntest
            print('rel_err : {}'.format(rel_err))
            if model_run is not None and model_run.recorder is not None:
                model_run.recorder.record_epoch(ep + 1, dict(train_loss=train_loss, validation_relative_l2=rel_err))
                model_run.recorder.visualize(model, ep + 1, args.epochs, dataset=test_loader.dataset, output_normalizer=y_normalizer)
            if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                model_run.complete_epoch(ep + 1)
            if ep % 100 == 0:
                if model_run is not None:
                    model_run.save(model)
                else:
                    if not os.path.exists('./checkpoints'):
                        os.makedirs('./checkpoints')
                    print('save model')
                    torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))
        if model_run is not None:
            model_run.save(model)
        else:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            print('save model')
            torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))
if __name__ == '__main__':
    main()
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        finish_linearno(args)
