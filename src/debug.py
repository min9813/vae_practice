import torch
import torch.nn as nn
from lib.network.vae import VAE
from lib.lossfunction.kl_div import kl_div_normal


def test_model():
    vae = VAE(3)
    x = torch.randn(1, 3, 32, 32)
    x_hat, mean_v, log_var = vae(x)
    kl = kl_div_normal(mean_v, log_var)
    rec_loss = nn.MSELoss()
    mse = rec_loss(x, x_hat)
    print(kl.size(), kl, mse)


if __name__ == "__main__":
    test_model()