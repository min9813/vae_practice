import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, padding=1, act="relu"):
        assert act in ("relu", "sigmoid")
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_ch)
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.bn(self.conv(x))
        h = self.act(h)
        return h


class ConvTransposeBlock(nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, padding=1, act="relu"):
        assert act in ("relu", "sigmoid")
        super(ConvTransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_ch)
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.bn(self.conv(x))
        h = self.act(h)
        return h


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 2, 2)


class Encoder(nn.Module):

    def __init__(self, input_ch, h_dim=1024, z_dim=64):

        super(Encoder, self).__init__()
        self.models = nn.Sequential(
            ConvBlock(input_ch, 32, 4, 2, 1),
            ConvBlock(32, 64, 4, 2, 1),
            ConvBlock(64, 128, 4, 2, 1),
            ConvBlock(128, 256, 4, 2, 1)
        )
        # self.conv1 = ConvBlock(input_ch, 32, 4, 2, 1)
        # self.conv2 = ConvBlock(32, 64, 4, 2, 1)
        # self.conv3 = ConvBlock(64, 128, 4, 2, 1)
        # self.conv4 = ConvBlock(128, 256, 4, 2, 1)

        self.flatten = Flatten()

        self.mean_fc = nn.Linear(h_dim, z_dim)
        self.log_sigma_fc = nn.Linear(h_dim, z_dim)

    def forward(self, x):

        h = self.models(x)
        h = self.flatten(h)
        mean_v = self.mean_fc(h)
        log_sigma_v = self.log_sigma_fc(h)

        return mean_v, log_sigma_v


class Decoder(nn.Module):

    def __init__(self, output_ch, h_dim=1024, z_dim=64):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, h_dim)
        self.unflatten = UnFlatten()

        self.models = nn.Sequential(
            ConvTransposeBlock(256, 128, 4, 2, 1),
            ConvTransposeBlock(128, 64, 4, 2, 1),
            ConvTransposeBlock(64, 32, 4, 2, 1),
            ConvTransposeBlock(32, output_ch, 4, 2, 1, act="sigmoid")
        )

    def forward(self, z):
        h = self.fc(z)
        h = self.unflatten(h)

        h = self.models(h)

        return h


class VAE(nn.Module):

    def __init__(self, image_ch, h_dim=1024, z_dim=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(image_ch, h_dim, z_dim)
        self.decoder = Decoder(image_ch, h_dim, z_dim)
        self.z_dim = z_dim

    def forward(self, x):
        mean_v, log_sigma_v = self.encoder(x)
        std_v = log_sigma_v.mul(0.5).exp_() # std = sqrt(var) <=> log std = 1/2 * log var 
        randn = self.genereate_random(mean_v.shape, x.dtype)
        
        z = randn * std_v + mean_v

        z = self.decoder(z)

        return z, mean_v, log_sigma_v

    def genereate_random(self, size, dtype):
        return torch.randn(size, dtype=dtype)


if __name__ == "__main__":
    vae = VAE(3)
    x = torch.randn(1, 3, 32, 32)
    x_hat, mean_v, std_v = vae(x)
    print(x_hat.size())
    print(mean_v.size())
    print(std_v.size())