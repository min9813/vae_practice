import os
import sys
import pathlib
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import logging
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import torch
import torch.nn as nn
import lib.utils.logger_config as logger_config
from lib.network.vae import VAE
from lib.lossfunction.kl_div import kl_div_normal
import lib.dataset.cifar as cifar
import lib.utils.get_aug_and_trans as get_aug_and_trans
from lib.utils.configuration import cfg as args
from lib.utils.configuration import cfg_from_file, format_dict

def fix_seed(seed=0):
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))


def test_model():
    vae = VAE(3)
    x = torch.randn(1, 3, 32, 32)
    x_hat, mean_v, log_var = vae(x)
    kl = kl_div_normal(mean_v, log_var)
    rec_loss = nn.MSELoss()
    mse = rec_loss(x, x_hat)
    print(kl.size(), kl, mse)


def train():
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        cfg_from_file(cfg_file)
    else:
        cfg_file = "default"
        print("!!!!! No specified configfile, so use default config !!!!!")

    if len(args.gpus.split(',')) > 1 and args.use_multi_gpu:
        multi_gpus = True
    else:
        multi_gpus = False
    args.multi_gpus = multi_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.is_cpu:
        print("cpu!!")
        args.device = torch.device("cpu")
    else:
        if args.multi_gpus:
            args.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.cuda_id)
            print("use cuda id:", args.device)

    fix_seed(args.seed)

    make_directory(args.LOG.save_dir)

    config_file = pathlib.Path(cfg_file)
    stem = config_file.stem
    args.exp_version = stem

    parent = config_file.parent.stem
    args.exp_type = parent

    args.model_save_dir = f"{args.LOG.save_dir}/{args.exp_type}/{args.exp_version}"
    args.image_save_dir = f"{args.LOG.save_image_dir}/{args.exp_type}/{args.exp_version}"

    msglogger = logger_config.config_pylogger(
        './config/logging.conf', args.exp_version, output_dir="{}/{}".format(args.LOG.save_dir, parent))
    trn_logger = logging.getLogger().getChild('train')
    val_logger = logging.getLogger().getChild('valid')

    msglogger.info("#"*30)
    msglogger.info("#"*5 + "\t" + "CONFIG FILE: " + str(config_file))

    msglogger.info("#"*30)

    if args.debug:
        args.TRAIN.total_epoch = 500
        args.LOG.train_print_iter = 1

    # args.TRAIN.fp16 = args.TRAIN.fp16 and fp16

    trans = get_aug_and_trans.get_aug_trans(None, None, None, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    trn_dataset = cifar.Cifar10("train", args, msglogger, trans)
    train_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    save_dir = "./check"
    make_directory(save_dir)
    for idx, data in enumerate(train_loader):
        image = data["data"][0]
        image = (image + 1.) * 0.5 * 255
        image = image.numpy()
        image = image.transpose(1, 2, 0)
        image = image.astype(np.uint8)
        path = os.path.join(save_dir, "{:0>5}.jpg".format(idx))
        cv2.imwrite(path, image)
        if idx > 10:
            break



if __name__ == "__main__":
    # test_model()
    train()