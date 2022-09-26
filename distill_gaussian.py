import os
import fire
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

import models
import datasets
import conventions
from utils import misc


criterion = misc.DistillationLoss()
xe = nn.CrossEntropyLoss(reduction='mean')

def distill_using_data(teacher_nw, student_nw, train_loader, valid_loader, n_epochs, lr, verbose, device, save, LOG_DIR):
    print("\nDistillation using original training data..")
    optimizer = Adam(student_nw.parameters(), lr=lr)
    metrics = []
    valid_loss_prev = np.inf
    for epoch in range(1, n_epochs+1):
        teacher_nw.eval()
        student_nw.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = teacher_nw(data)
            student_output = student_nw(data)
            loss = criterion(student_output, teacher_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        student_nw.eval()
        valid_loss = 0.0
        accs = []
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                student_output = student_nw(data)
            loss = xe(student_output, target)
            valid_loss += loss.item()
            accs.append(misc.accuracy_metric(student_output.detach(), target))

        metrics.append([train_loss/len(train_loader), valid_loss/len(valid_loader), np.mean(accs)])
        if verbose and epoch % verbose==0:
            print("Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(
                epoch, *metrics[-1]))

    return [list(i) for i in zip(*metrics)]  # train_loss, valid_loss, valid_acc


def distill_using_noise(model_family, teacher_nw, student_nw, valid_loader, n_epochs, len_batch, lr, verbose, device, save, LOG_DIR):
    print("\nDistillation using Gaussian noise..")
    optimizer = Adam(student_nw.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    metrics = []

    data_sample = iter(valid_loader).next()[0]
    valid_loss_prev = np.inf
    teacher_nw.train()
    student_nw.train()
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        for batch in range(len_batch):
            optimizer.zero_grad()
            data = torch.randn_like(data_sample, device=device)
            with torch.no_grad():
                teacher_output = teacher_nw(data)
            student_output = student_nw(data)
            loss = criterion(student_output, teacher_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        accs = []
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                student_output = student_nw(data)
            loss = xe(student_output, target)
            valid_loss += loss.item()
            accs.append(misc.accuracy_metric(student_output.detach(), target))

        scheduler.step(valid_loss)

        metrics.append([train_loss/len_batch, valid_loss/len(valid_loader), np.mean(accs)])
        if verbose and epoch % verbose==0:
            print("Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(
                epoch, *metrics[-1]))
            with open(LOG_DIR + '/noise_log.txt', 'a') as f:
                line = "Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(
                    epoch, *metrics[-1])
                f.write(line + '\n')

    return [list(i) for i in zip(*metrics)]  # train_loss, valid_loss, valid_acc


@misc.log_experiment
def experiment_distil_gaussian(dataset_name, n_epochs_gaussian, n_epochs_data, lr=1e-3, compare=False, verbose=True, LOG_DIR='./', desc=None, save=False, **kwargs):
    print("Description:", desc)
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    model_family = experiment_config['model_teacher']      


    print('Experiment Configuration:')
    print(experiment_config)

    train_loader, _, valid_loader = eval("datasets.get_{}({})".format(
        dataset_name,
        experiment_config['batch_size']
    ))

    len_batch = len(train_loader)

    teacher_name = conventions.resolve_teacher_name(experiment_config)
    teacher_path = os.path.join("Pretrained_NW", teacher_name)
    teacher_nw = torch.load(teacher_path)
    teacher_nw.to(device)

    student_nw = eval("models.{}.Target_Net({}, {})".format(
        experiment_config['model_student'],
        experiment_config['inputs'],
        experiment_config['code_dim']
    )).to(device)

    metrics = distill_using_noise(model_family, teacher_nw, student_nw, valid_loader, n_epochs_gaussian, len_batch, lr, verbose, device, save, LOG_DIR)
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Accuracy Noise")

    if compare:
        teacher_nw = torch.load(teacher_path)
        student_nw = student_nw.apply(misc.weight_reset)
        metrics = distill_using_data(teacher_nw, student_nw, train_loader, valid_loader, n_epochs_data, lr, verbose, device, save, LOG_DIR)
        plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Accuracy Data")

    plt.title('Student Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy.png'), dpi=400)
    plt.close()


if __name__ == '__main__':
    fire.Fire(experiment_distil_gaussian)
