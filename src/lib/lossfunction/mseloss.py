import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mseloss(y_hat, y):
    diff = y_hat - y
    loss = torch.sum(torch.mean(diff * diff, dim=0))
    return loss
