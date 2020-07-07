import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div_normal(q_mean, q_log_var):
    q_var = q_log_var.exp()
    # print(q_var.mean(), (q_mean*q_mean).mean(), q_log_var.mean())
    kl = -0.5 * torch.mean(torch.sum(1 + q_log_var - q_mean*q_mean - q_var, dim=1))

    return kl
