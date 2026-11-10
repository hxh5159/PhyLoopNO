import os
import argparse
from cdlno_entry import parse_args as parse_task_args
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser('Training Transolver')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_2D')
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
parser.add_argument('--save_name', type=str, default='darcy_Transolver')
parser.add_argument('--data_path', type=str, default='/data/fno')
args = parse_task_args(parser, 'darcy')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
    from linearno_entry import model_kwargs as linearno_model_kwargs, StandardRun as LinearNORun
    from linearno_entry import start as start_linearno, finish as finish_linearno
    from linearno_entry import normalizer as linearno_normalizer, verify_data as verify_linearno_data
    start_linearno(args, 'darcy')
train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
ntrain = args.ntrain
ntest = 200
epochs = args.epochs
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

def central_diff(x: torch.Tensor, h, resolution):
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0.0)
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)
    return (grad_x, grad_y)

def main():
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        verify_linearno_data(args)
    r = args.downsample
    h = int((421 - 1) / r + 1)
    s = h
    dx = 1.0 / s
    train_data = scio.loadmat(train_path)
    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
    x_train = x_train.reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float()
    y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train.reshape(ntrain, -1)
    y_train = torch.from_numpy(y_train)
    test_data = scio.loadmat(test_path)
    x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
    x_test = x_test.reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float()
    y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
    y_test = y_test.reshape(ntest, -1)
    y_test = torch.from_numpy(y_test)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        x_normalizer = linearno_normalizer(args, 'input', x_train, UnitTransformer)
    else:
        x_normalizer = UnitTransformer(x_train)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        y_normalizer = linearno_normalizer(args, 'output', y_train, UnitTransformer)
    else:
        y_normalizer = UnitTransformer(y_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)
    x_normalizer.cuda()
    y_normalizer.cuda()
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print('Dataloading is over.')
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train), batch_size=args.batch_size, shuffle=True)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test), batch_size=args._linearno_config['values']['training']['test_batch_size'], shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test), batch_size=args.batch_size, shuffle=False)
    model_run = None
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        model = get_model(args).Model(**linearno_model_kwargs(args, H=s, W=s)).cuda()
        model_run = LinearNORun(args, model)
        model_run.recorder.update_protocol(dict(ntrain=ntrain, ntest=ntest))
    else:
        model = get_model(args).Model(space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden, dropout=args.dropout, n_head=args.n_heads, Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=1, out_dim=1, slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos, H=s, W=s).cuda()
    results_dir = model_run.result_dir if model_run is not None else './results/' + save_name + '/'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs, steps_per_epoch=len(train_loader))
    if model_run is not None and model_run.recorder is not None:
        model_run.recorder.record_training_setup(optimizer, scheduler)
    myloss = TestLoss(size_average=False)
    de_x = TestLoss(size_average=False)
    de_y = TestLoss(size_average=False)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        model_run.prepare(optimizer, scheduler, train_loader, test_loader)
        myloss = TestLoss(size_average=True)
        de_x = TestLoss(size_average=True)
        de_y = TestLoss(size_average=True)
    if eval:
        print('model evaluation')
        print(s, s)
        if model_run is not None:
            model_run.load(model)
        else:
            model.load_state_dict(torch.load('./checkpoints/' + save_name + '.pt'), strict=False)
        model.eval()
        showcase = 10
        id = 0
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with torch.no_grad():
            rel_err = 0.0
            with torch.no_grad():
                for x, fx, y in test_loader:
                    id += 1
                    x, fx, y = (x.cuda(), fx.cuda(), y.cuda())
                    out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                        rel_err += tl * y.shape[0]
                    else:
                        rel_err += tl
                    if id < showcase:
                        print(id)
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(out[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(os.path.join(results_dir, 'case_' + str(id) + '_pred.pdf'))
                        plt.close()
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(y[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(os.path.join(results_dir, 'case_' + str(id) + '_gt.pdf'))
                        plt.close()
                        plt.figure()
                        plt.axis('off')
                        plt.imshow((y[0, :] - out[0, :]).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.clim(-0.0005, 0.0005)
                        plt.savefig(os.path.join(results_dir, 'case_' + str(id) + '_error.pdf'))
                        plt.close()
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(fx[0, :].unsqueeze(-1).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(os.path.join(results_dir, 'case_' + str(id) + '_input.pdf'))
                        plt.close()
            rel_err /= ntest
            print('rel_err:{}'.format(rel_err))
            if model_run is not None and model_run.recorder is not None:
                model_run.recorder.record_metrics(dict(relative_l2=rel_err))
    else:
        for ep in range(args.epochs):
            if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                if ep < model_run.start_epoch:
                    continue
            model.train()
            train_loss = 0
            reg = 0
            for x, fx, y in train_loader:
                x, fx, y = (x.cuda(), fx.cuda(), y.cuda())
                optimizer.zero_grad()
                out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                l2loss = myloss(out, y)
                out = rearrange(out.unsqueeze(-1), 'b (h w) c -> b c h w', h=s)
                out = out[..., 1:-1, 1:-1].contiguous()
                out = F.pad(out, (1, 1, 1, 1), 'constant', 0)
                out = rearrange(out, 'b c h w -> b (h w) c')
                gt_grad_x, gt_grad_y = central_diff(y.unsqueeze(-1), dx, s)
                pred_grad_x, pred_grad_y = central_diff(out, dx, s)
                deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y, gt_grad_y)
                loss = 0.1 * deriv_loss + l2loss
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                    train_loss += l2loss.item() * y.shape[0]
                else:
                    train_loss += l2loss.item()
                if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                    reg += deriv_loss.item() * y.shape[0]
                else:
                    reg += deriv_loss.item()
                scheduler.step()
            train_loss /= ntrain
            reg /= ntrain
            print('Epoch {} Reg : {:.5f} Train loss : {:.5f}'.format(ep, reg, train_loss))
            model.eval()
            rel_err = 0.0
            id = 0
            with torch.no_grad():
                for x, fx, y in test_loader:
                    id += 1
                    if id == 2:
                        vis = True
                    else:
                        vis = False
                    x, fx, y = (x.cuda(), fx.cuda(), y.cuda())
                    out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                        rel_err += tl * y.shape[0]
                    else:
                        rel_err += tl
            rel_err /= ntest
            print('rel_err:{}'.format(rel_err))
            if model_run is not None and model_run.recorder is not None:
                model_run.recorder.record_epoch(ep + 1, dict(train_loss=train_loss, validation_relative_l2=rel_err, derivative_regularizer=reg))
                model_run.recorder.visualize(model, ep + 1, args.epochs, dataset=test_loader.dataset, grid_shape=(s, s), output_normalizer=y_normalizer)
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
