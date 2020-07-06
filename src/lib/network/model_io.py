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
import torch
try:
    from apex import amp
except ImportError:
    fp16 = False


def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))


def load_model(net, path, logger=None):
    if logger is None:
        print_ = print
    else:
        print_ = logger.info
    print_("Load model from {}".format(path))
    state_dict = torch.load(path)
    score = state_dict["score"]
    epoch = state_dict["iters"]
    if isinstance(score, dict):
        msg = ""
        for key, value in state_dict["score"].items():
            msg += " {}:{} ".format(key, value)
    else:
        msg = str(score)
    print_(f"Best score :{score}, iter:{epoch}")

    return net, epoch+1


def save_model(wrapper, optimizer, score, is_best, epoch, logger=None, multi_gpus=False, model_save_dir="../models", delete_old=False, is_amp=False):
    if logger is None:
        print_ = print
    else:
        print_ = logger.info
    msg = 'Saving checkpoint: {}'.format(epoch)
    print_(msg)
    if multi_gpus:
        model = wrapper.module.model.state_dict()

    else:
        model = wrapper.model.state_dict()

    model_save_dir = model_save_dir
    make_directory(model_save_dir)
    old_model_path = list(pathlib.Path(
        model_save_dir).glob("latest*.ckpt"))
    save_net_path = os.path.join(
        model_save_dir, 'latest_{:0>6}_amp_net.ckpt'.format(epoch))
    save_best_net_path = os.path.join(
        model_save_dir, 'best_amp_net.ckpt'.format(epoch))
    save_other_path = os.path.join(
        model_save_dir, 'latest_{:0>6}_opt.ckpt'.format(epoch))
    save_best_other_path = os.path.join(
        model_save_dir, 'best_opt.ckpt'.format(epoch))

    torch.save({
        'iters': epoch,
        'net': model,
        'score': score
    },
        save_net_path)
    save_dict = {
        'iters': epoch,
        'optimizer': optimizer.state_dict(),
        # 'amp': amp.state_dict()
    }

    if fp16 and is_amp:
        save_dict["amp"] = amp.state_dict()
    # if args.fp16_train:
    #     save_dict["amp"] = amp.state_dict()
    torch.save(
        save_dict,
        save_other_path)

    if is_best:
        shutil.copy(str(save_other_path),
                    str(save_best_other_path))
        shutil.copy(str(save_net_path), str(save_best_net_path))

    if delete_old:
        for p in old_model_path:
            os.remove(str(p))
