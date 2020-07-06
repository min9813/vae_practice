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
import torch.utils.data as data


class Cifar10(data.Dataset):

    def __init__(self, split, args, logger, trans=None, c_aug=None, s_aug=None):
        assert split in ("train", "test")
        self.args = args
        self.logger = logger
        self.split = split

        self.trans = trans
        self.c_aug = c_aug
        self.s_aug = s_aug

        self.logger.info(f"setup cifar {split} ==>")
        cifar_data = self.setup_cifar()
        self.dataset = cifar_data

    def setup_cifar(self):
        """
        args.cifar_root_dir
        """
        # load meta data
        meta_data_path = os.path.join(
            self.args.DATA.cifar_root_dir, self.args.DATA.cifar_meta_file)
        with open(meta_data_path, "rb") as pkl:
            meta_data = pickle.load(pkl)

        if self.split == "train":
            path_list = pathlib.Path(self.args.DATA.cifar_root_dir).glob(
                self.args.DATA.train_reg_exp)
        else:
            path_list = pathlib.Path(self.args.DATA.cifar_root_dir).glob(
                self.args.DATA.test_reg_exp)

        all_data = []
        all_labels = []
        for path in path_list:
            with open(str(path), "rb") as pkl:
                this_batch = pickle.load(pkl, encoding="bytes")
            """
            loaded data is (batch, 3*32*32), 3=(RGB)
            """
            use_keys = list(this_batch.keys())
            for key in use_keys:
                if isinstance(key, str):
                    continue
                this_batch[key.decode("ascii")] = this_batch[key]
                this_batch.pop(key)

            images = this_batch["data"].reshape(-1, 3, 32, 32)
            all_data.append(images)
            all_labels.append(this_batch["labels"])
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_data, axis=0)

        cifar_data = {
            "data": all_data,
            "labels": all_labels,
            "meta_data": meta_data
        }

        return cifar_data

    def __len__(self):
        return len(self.dataset["data"])

    def pickup(self, index):
        data = self.dataset["data"][index]
        label = self.dataset["labels"][index]
        data = data.transpose(1, 2, 0)
        if self.trans is not None:
            data = self.trans(data)

        data = {"data": data, "label": label,
                "label_name": self.dataset["meta_data"]["label_names"][label]}

        return data

    def __getitem__(self, index):
        picked_data = self.pickup(index)

        return picked_data
