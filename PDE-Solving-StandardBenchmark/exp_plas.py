import os
import argparse
from cdlno_entry import parse_args as parse_task_args
import matplotlib.pyplot as plt
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
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='plas_Transolver')
parser.add_argument('--data_path', type=str, default='/data/fno/plas_N987_T20.mat')
args = parse_task_args(parser, 'plasticity')
eval = args.eval
save_name = args.save_name
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
    from linearno_entry import model_kwargs as linearno_model_kwargs, StandardRun as LinearNORun
    from linearno_entry import start as start_linearno, finish as finish_linearno
    from linearno_entry import normalizer as linearno_normalizer, verify_data as verify_linearno_data
    start_linearno(args, 'plasticity')
import numpy as np
import scipy.io as scio
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f'Total Trainable Params: {total_params}')
    return total_params

def random_collate_fn(batch):
    shuffled_batch = []
    shuffled_u = None
    shuffled_t = None
    shuffled_a = None
    shuffled_pos = None
    for item in batch:
        pos = item[0]
        t = item[1]
        a = item[2]
        u = item[3]
        num_timesteps = t.size(0)
        permuted_indices = torch.randperm(num_timesteps)
        t = t[permuted_indices]
        u = u[..., permuted_indices]
        if shuffled_t is None:
            shuffled_pos = pos.unsqueeze(0)
            shuffled_t = t.unsqueeze(0)
            shuffled_u = u.unsqueeze(0)
            shuffled_a = a.unsqueeze(0)
        else:
            shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
            shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
            shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
            shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)
    shuffled_batch.append(shuffled_pos)
    shuffled_batch.append(shuffled_t)
    shuffled_batch.append(shuffled_a)
    shuffled_batch.append(shuffled_u)
    return shuffled_batch

def main():
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        verify_linearno_data(args)
    DATA_PATH = args.data_path
    N = 987
    ntrain = 900
    ntest = 80
    s1 = 101
    s2 = 31
    T = 20
    Deformation = 4
    r1 = 1
    r2 = 1
    s1 = int((s1 - 1) / r1 + 1)
    s2 = int((s2 - 1) / r2 + 1)
    data = scio.loadmat(DATA_PATH)
    input = torch.tensor(data['input'], dtype=torch.float)
    output = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)
    print(input.shape, output.shape)
    x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2)
    x_train = x_train.reshape(ntrain, -1, 1)
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = y_train.reshape(ntrain, -1, Deformation, T)
    x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2)
    x_test = x_test.reshape(ntest, -1, 1)
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = y_test.reshape(ntest, -1, Deformation, T)
    print(x_train.shape, y_train.shape)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        x_normalizer = linearno_normalizer(args, 'input', x_train, UnitTransformer)
    else:
        x_normalizer = UnitTransformer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    x_normalizer.cuda()
    x = np.linspace(0, 1, s1)
    y = np.linspace(0, 1, s2)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print('Dataloading is over.')
    t = np.linspace(0, 1, T)
    t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
    t_train = t.repeat(ntrain, 1)
    t_test = t.repeat(ntest, 1)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train), batch_size=args.batch_size, shuffle=True, collate_fn=random_collate_fn)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test), batch_size=args._linearno_config['values']['training']['test_batch_size'], shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test), batch_size=args.batch_size, shuffle=False)
    print('Dataloading is over.')
    model_run = None
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        model = get_model(args).Model(**linearno_model_kwargs(args, H=s1, W=s2)).cuda()
        model_run = LinearNORun(args, model)
    else:
        model = get_model(args).Model(space_dim=2, n_hidden=args.n_hidden, n_layers=args.n_layers, Time_Input=True, n_head=args.n_heads, fun_dim=1, out_dim=Deformation, mlp_ratio=args.mlp_ratio, slice_num=args.slice_num, unified_pos=args.unified_pos, H=s1, W=s2).cuda()
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
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        test_l2_step = 0
        test_l2_full = 0
        showcase = 10
        id = 0
        with torch.no_grad():
            for x, tim, fx, yy in test_loader:
                id += 1
                loss = 0
                x, fx, tim, yy = (x.cuda(), fx.cuda(), tim.cuda(), yy.cuda())
                bsz = x.shape[0]
                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    im = model(x, fx, T=input_T)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im.unsqueeze(-1)
                    else:
                        pred = torch.cat((pred, im.unsqueeze(-1)), -1)
                if id < showcase:
                    print(id)
                    truth = y[0].reshape(101, 31, 4).squeeze().detach().cpu().numpy()
                    pred_vis = im[0].reshape(101, 31, 4).squeeze().detach().cpu().numpy()
                    truth_du = np.linalg.norm(truth[:, :, 2:], axis=-1)
                    pred_du = np.linalg.norm(pred_vis[:, :, 2:], axis=-1)
                    plt.axis('off')
                    plt.scatter(truth[:, :, 0], truth[:, :, 1], 10, truth_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 6)
                    plt.savefig(os.path.join(results_dir, 'gt_' + str(id) + '.pdf'), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.scatter(pred_vis[:, :, 0], pred_vis[:, :, 1], 10, pred_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 6)
                    plt.savefig(os.path.join(results_dir, 'pred_' + str(id) + '.pdf'), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.scatter(truth[:, :, 0], truth[:, :, 1], 10, pred_du[:, :] - truth_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.2, 0.2)
                    plt.savefig(os.path.join(results_dir, 'error_' + str(id) + '.pdf'), bbox_inches='tight', pad_inches=0)
                    plt.close()
                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
        print('test_step_loss:{:.5f} , test_full_loss:{:.5f}'.format(test_l2_step / ntest / T, test_l2_full / ntest))
        if model_run is not None and model_run.recorder is not None:
            model_run.recorder.record_metrics(dict(test_step_loss=test_l2_step / ntest / T, test_full_loss=test_l2_full / ntest))
    else:
        for ep in range(args.epochs):
            if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
                if ep < model_run.start_epoch:
                    continue
            model.train()
            train_l2_step = 0
            for x, tim, fx, yy in train_loader:
                x, fx, tim, yy = (x.cuda(), fx.cuda(), tim.cuda(), yy.cuda())
                bsz = x.shape[0]
                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    im = model(x, fx, T=input_T)
                    loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    train_l2_step += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
            model.eval()
            test_l2_step = 0
            test_l2_full = 0
            with torch.no_grad():
                for x, tim, fx, yy in test_loader:
                    loss = 0
                    x, fx, tim, yy = (x.cuda(), fx.cuda(), tim.cuda(), yy.cuda())
                    bsz = x.shape[0]
                    for t in range(T):
                        y = yy[..., t:t + 1]
                        input_T = tim[:, t:t + 1].reshape(bsz, 1)
                        im = model(x, fx, T=input_T)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im.unsqueeze(-1)
                        else:
                            pred = torch.cat((pred, im.unsqueeze(-1)), -1)
                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            print('Epoch {} , train_step_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}'.format(ep, train_l2_step / ntrain / T, test_l2_step / ntest / T, test_l2_full / ntest))
            if model_run is not None and model_run.recorder is not None:
                model_run.recorder.record_epoch(ep + 1, dict(train_step_loss=train_l2_step / ntrain / T, test_step_loss=test_l2_step / ntest / T, test_full_loss=test_l2_full / ntest))
                model_run.recorder.visualize(model, ep + 1, args.epochs, dataset=test_loader.dataset, grid_shape=(s1, s2))
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
