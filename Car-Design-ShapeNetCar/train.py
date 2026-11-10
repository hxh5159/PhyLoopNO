import numpy as np
import time, json, os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def get_nb_trainable_params(model):
    """
    Return the number of trainable parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def train(device, model, train_loader, optimizer, scheduler, reg=1):
    model.train()
    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in train_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        optimizer.zero_grad()
        out = model((cfd_data, geom))
        targets = cfd_data.y
        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return (np.mean(losses_press), np.mean(losses_velo))

@torch.no_grad()
def test(device, model, test_loader):
    model.eval()
    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y
        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return (np.mean(losses_press), np.mean(losses_velo))

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(device, train_dataset, val_dataset, Net, hparams, path, reg=1, val_iter=1, coef_norm=[], record=None, linearno_run=None):
    if linearno_run is not None:
        return linearno_run.train(device, train_dataset, val_dataset, Net, hparams, path, reg, val_iter, coef_norm, record)
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['lr'], total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'], final_div_factor=1000.0)
    if record is not None:
        record.record_training_setup(optimizer, lr_scheduler)
    start = time.time()
    train_loss, val_loss = (100000.0, 100000.0)
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        loss_velo, loss_press = train(device, model, train_loader, optimizer, lr_scheduler, reg=reg)
        if record is not None:
            recorded_train = dict(pressure_mse=loss_velo, velocity_mse=loss_press)
        train_loss = loss_velo + reg * loss_press
        del train_loader
        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
            val_loader = DataLoader(val_dataset, batch_size=1)
            loss_velo, loss_press = test(device, model, val_loader)
            val_loss = loss_velo + reg * loss_press
            del val_loader
            pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
        else:
            pbar_train.set_postfix(train_loss=train_loss)
        if record is not None:
            metrics = dict(train_components=recorded_train, upstream_train_log_value=train_loss)
            if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
                metrics.update(validation_pressure_mse=loss_velo, validation_velocity_mse=loss_press, upstream_validation_log_value=val_loss)
            record.record_epoch(epoch + 1, metrics)
            record.visualize(model, epoch + 1, hparams['nb_epochs'], dataset=val_dataset, coef_norm=coef_norm)
    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model, path + os.sep + f"model_{hparams['nb_epochs']}.pth")
    if val_iter is not None:
        with open(path + os.sep + f"log_{hparams['nb_epochs']}.json", 'a') as f:
            json.dump({'nb_parameters': params_model, 'time_elapsed': time_elapsed, 'hparams': hparams, 'train_loss': train_loss, 'val_loss': val_loss, 'coef_norm': list(coef_norm)}, f, indent=12, cls=NumpyEncoder)
    return model
