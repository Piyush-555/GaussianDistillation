import os
import fire
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F

import models
import datasets
import conventions
from utils import misc


def train_one_epoch(target_nw, train_loader, valid_loader, optimizer, criterion, scheduler, device):
    target_nw.train()
    train_loss = 0.0
    accs = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = target_nw(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        accs.append(misc.accuracy_metric(output.detach(), target))
    train_acc = np.mean(accs)
    
    target_nw.eval()
    valid_loss = 0.0
    accs = []
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = target_nw(data)
        loss = criterion(output, target)
        valid_loss += loss.item()
        accs.append(misc.accuracy_metric(output.detach(), target))
    valid_acc = np.mean(accs)

    scheduler.step(valid_loss)
    
    return train_loss/len(train_loader), train_acc, valid_loss/len(valid_loader), valid_acc


def train_teacher(target_nw, train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(target_nw.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    metrics = []
    valid_loss_prev = np.inf
    for epoch in range(1, n_epochs+1):
        args = train_one_epoch(target_nw, train_loader, valid_loader, optimizer, criterion, scheduler, device)
        if verbose:
            print("Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(epoch, *args))
        metrics.append(args)

        if args[-2] < valid_loss_prev and save:
            print('Saving teacher...')
            torch.save(target_nw, "teacher.model")
            valid_loss_prev = args[-2]

    return [list(i) for i in zip(*metrics)]  # train_loss, train_acc, valid_loss, valid_acc


@misc.log_experiment
def util_train_teacher(dataset_name, n_epochs, lr=1e-3, weight_decay=0, verbose=True, save=False, LOG_DIR='./', **kwargs):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs('data', exist_ok=True)
    os.makedirs('Pretrained_NW', exist_ok=True)

    train_loader, _, valid_loader = eval("datasets.get_{}({})".format(
        dataset_name,
        experiment_config['batch_size']
    ))

    model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config['model_teacher'],
        experiment_config['inputs'],
        experiment_config['code_dim']
    )).to(device)

    metrics = train_teacher(model, train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR)

    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    plt.title('Teacher Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Teacher Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss.png'), dpi=200)
    plt.close()

    model_name = conventions.resolve_teacher_name(experiment_config)
    torch.save(model, os.path.join('Pretrained_NW', model_name))


if __name__ == '__main__':
    fire.Fire(util_train_teacher)
