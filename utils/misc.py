import os
import sys
import random
import datetime
import traceback
import contextlib

import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F


################################################################
# General
################################################################

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy_metric(outputs, targets):
    return np.mean(outputs.detach().cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or \
        isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()


class DistillationLoss:  # Following name is more appropriate
    def __init__(self, T=1):
        self.T = T

    def __call__(self, student, teacher):
        student = F.log_softmax(student/self.T, dim=-1)
        teacher = (teacher/self.T).softmax(dim=-1)
        
        try: return -(teacher * student).sum(dim=1).mean()
        except: import pdb; pdb.set_trace()


################################################################
# Logging
################################################################

@contextlib.contextmanager
def print_to_logfile(file):
    # Capture all outputs to a log file while still printing it
    class Logger:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def __getattr__(self, attr):
            return getattr(self.terminal, attr)

    logger = Logger(file)

    _stdout = sys.stdout
    sys.stdout = logger
    try:
        yield logger.log
    except:
        error = traceback.format_exc()
        logger.terminal.write(error)
        logger.log.write(error)
    finally:
        sys.stdout = _stdout


def log_experiment(experiment):
    # A decorator to log everything to a file
    def decorator(*args, **kwargs):
        time_now = datetime.datetime.now().strftime("%d %B %Y, %I.%M%p")
        experiment_dir = "Logs/{}".format(experiment.__name__)
        log_dir = os.path.join(experiment_dir, time_now)
        plots_dir = os.path.join(log_dir, "Plots")
        log_file = os.path.join(log_dir, "log.txt")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        with print_to_logfile(open(log_file, 'w')):
            print("Performing experiment:", experiment.__name__)
            print("Date-Time:", time_now)
            print("\n", end="")
            print("Args:", args)
            print("Kwargs:", kwargs)
            print("\n", end="")
            experiment(*args, **kwargs, LOG_DIR=log_dir)
            print("\nSuccessfully Executed!!")
    return decorator
