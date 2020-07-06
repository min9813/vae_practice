import torch
import torch.nn as nn
import torch.nn.functional as F


class LossWrap(torch.nn.Module):
    def __init__(self, args, model, rec_loss, kl_loss):
        self.args = args
        super(LossWrap, self).__init__()
        self.model = model
        self.rec_loss = rec_loss
        self.kl_loss = kl_loss

    def forward(self, input, label=None):
        if self.args.multi_gpus:
            if label is None:
                input = input.cuda()
            else:
                input, label = input.cuda(), label.cuda()
        else:
            if label is None:
                input = input.to(self.args.device)
            else:
                input, label = input.to(
                    self.args.device), label.to(self.args.device)

        x_hat, mean, log_var = self.model(input)
        kl_loss = self.kl_loss(mean, log_var)
        mse_loss = self.rec_loss(input, x_hat)

        return kl_loss, mse_loss
