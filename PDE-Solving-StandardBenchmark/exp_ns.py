import os
import matplotlib.pyplot as plt
import argparse
from cdlno_entry import parse_args as parse_task_args
import scipy.io as scio
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
parser = argparse.ArgumentParser('Training Transformer')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--gpu', type=str, default='0', help='GPU index to use')
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='ns_2d_UniPDE')
parser.add_argument('--data_path', type=str, default='/data/fno')
args = parse_task_args(parser, 'ns')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
    from linearno_entry import model_kwargs as linearno_model_kwargs, StandardRun as LinearNORun
    from linearno_entry import start as start_linearno, finish as finish_linearno
    from linearno_entry import normalizer as linearno_normalizer, verify_data as verify_linearno_data
    start_linearno(args, 'ns')
data_path = args.data_path + '/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat'
ntrain = 1000
ntest = 200
T_in = 10
T = 10
step = 1
eval = args.eval
save_name = args.save_name

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
    r = args.downsample
    h = int((64 - 1) / r + 1)
    data = scio.loadmat(data_path)
    print(data['u'].shape)
    train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
    train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
    train_a = torch.from_numpy(train_a)
    train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)
    test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
    test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
    test_a = torch.from_numpy(test_a)
    test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)
    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u), batch_size=args.batch_size, shuffle=True)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u), batch_size=args._linearno_config['values']['training']['test_batch_size'], shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u), batch_size=args.batch_size, shuffle=False)
    print('Dataloading is over.')
    model_run = None
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        model = get_model(args).Model(**linearno_model_kwargs(args, H=h, W=h)).cuda()
        model_run = LinearNORun(args, model)
    else:
        model = get_model(args).Model(space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden, dropout=args.dropout, n_head=args.n_heads, Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=T_in, out_dim=1, slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos, H=h, W=h).cuda()
    results_dir = model_run.result_dir if model_run is not None else './results/' + save_name + '/'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    if model_run is not None and model_run.recorder is not None:
        model_run.recorder.record_training_setup(optimizer, scheduler)
    myloss = TestLoss(size_average=False)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        model_run.prepare(optimizer, scheduler, train_loader, test_loader)
    if eval:
        if model_run is not None:
            model_run.load(model)
        else:
            model.load_state_dict(torch.load('./checkpoints/' + save_name + '.pt'), strict=False)
        model.eval()
        showcase = 10
        id = 0
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        test_l2_full = 0
        with torch.no_grad():
            for x, fx, yy in test_loader:
                id += 1
                x, fx, yy = (x.cuda(), fx.cuda(), yy.cuda())
                bsz = x.shape[0]
                for t in range(0, T, step):
                    im = model(x, fx=fx)
                    fx = torch.cat((fx[..., step:], im), dim=-1)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                if id < showcase:
                    print(id)
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(im[0, :, 0].reshape(64, 64).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-3, 3)
                    plt.savefig(os.path.join(results_dir, 'case_' + str(id) + '_pred_' + str(20) + '.pdf'))
                    plt.close()
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(yy[0, :, t].reshape(64, 64).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-3, 3)
                    plt.savefig(os.path.join(results_dir, 'case_' + str(id) + '_gt_' + str(20) + '.pdf'))
                    plt.close()
                    plt.figure()
                    plt.axis('off')
                    plt.imshow((im[0, :, 0].reshape(64, 64) - yy[0, :, t].reshape(64, 64)).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-2, 2)
                    plt.savefig(os.path.join(results_dir, 'case_' + str(id) + '_error_' + str(20) + '.pdf'))
                    plt.close()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            print(test_l2_full / ntest)
            if model_run is not None and model_run.recorder is not None:
                model_run.recorder.record_metrics(dict(test_full_loss=test_l2_full / ntest))
    else:
        for ep in range(args.epochs):
            if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                if ep < model_run.start_epoch:
                    continue
            model.train()
            train_l2_step = 0
            train_l2_full = 0
            for x, fx, yy in train_loader:
                loss = 0
                x, fx, yy = (x.cuda(), fx.cuda(), yy.cuda())
                bsz = x.shape[0]
                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(x, fx=fx)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    fx = torch.cat((fx[..., step:], y), dim=-1)
                train_l2_step += loss.item()
                train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
            test_l2_step = 0
            test_l2_full = 0
            model.eval()
            with torch.no_grad():
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = (x.cuda(), fx.cuda(), yy.cuda())
                    bsz = x.shape[0]
                    for t in range(0, T, step):
                        y = yy[..., t:t + step]
                        im = model(x, fx=fx)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)
                        fx = torch.cat((fx[..., step:], im), dim=-1)
                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            print('Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}'.format(ep, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step), test_l2_full / ntest))
            if model_run is not None and model_run.recorder is not None:
                model_run.recorder.record_epoch(ep + 1, dict(train_step_loss=train_l2_step / ntrain / (T / step), test_step_loss=test_l2_step / ntest / (T / step), test_full_loss=test_l2_full / ntest, train_full_loss=train_l2_full / ntrain))
                model_run.recorder.visualize(model, ep + 1, args.epochs, dataset=test_loader.dataset, grid_shape=(h, h))
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
