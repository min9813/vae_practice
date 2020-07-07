import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

HOME = os.environ["HOME"]
__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.DATA = edict()
__C.TRAIN = edict()
__C.OPTIM = edict()
__C.TEST = edict()
__C.LOG = edict()
__C.MODEL = edict()

__C.DATA.batch_size = 64
__C.DATA.cifar_root_dir = ""
__C.DATA.cifar_meta_file = ""
__C.DATA.cifar_train_reg_exp = ""
__C.DATA.cifar_test_reg_exp = ""

__C.TRAIN.total_epoch = 100
__C.TRAIN.start_epoch = 1
__C.TRAIN.rec_loss = "mse"
__C.TRAIN.fp16 = False

__C.OPTIM.optimizer = "adam"
__C.OPTIM.lr = 1e-4
__C.OPTIM.lr_scheduler = "no"
__C.OPTIM.lr_milestones = [1000000]
__C.OPTIM.lr_gamma = 0.1
__C.OPTIM.lr_reduce_mode = "max"
__C.OPTIM.lr_patience = 5
__C.OPTIM.lr_min = 1e-6
__C.OPTIM.lr_cooldown = 1
__C.OPTIM.lr_tmax = 1

__C.LOG.save_dir = "../logs"
__C.LOG.save_image_dir = "../result/images"
__C.LOG.train_print_iter = 200
__C.LOG.train_print = True

__C.MODEL.save_dir = "../models"
__C.MODEL.delete_old = True
__C.MODEL.resume_net_path = ""
__C.MODEL.resume_opt_path = ""
__C.MODEL.resume = False
__C.MODEL.h_dim = 1024
__C.MODEL.z_dim = 64

__C.debug = True
__C.run_mode = "train"
__C.seed = 1234
__C.gpus = "0"
__C.use_multi_gpu = False
__C.is_cpu = False
__C.cuda_id = 0
__C.num_workers = 4
__C.input_ch = 3

def format_dict(cfg):
    ng_names = []
    key_list = set(list(cfg.keys()))
    for key1 in key_list:
        for key2 in key_list:
            if key1 == key2:
                continue
            value1 = cfg[key1]
            value2 = cfg[key2]
            if not isinstance(value1, edict):
                continue
            if not isinstance(value2, edict):
                continue
            for key_in_value1 in value1:
                if key_in_value1 in value2.keys():
                    ng_names.appeend((key_in_value1, key1, key2))
                    continue
                if key_in_value1 in key_list:
                    ng_names.append((key_in_value1, key1, "root"))
                    continue

    ng_names = list(set(ng_names))

    if len(ng_names):
        msg = ""
        for name in ng_names:
            msg += f"{name[0]} in ({name[1]},{name[2]})\n"
        raise ValueError(f"Same key can\'t exist in different dictionary \n{msg}")

    for key1 in key_list:
        if isinstance(cfg[key1], edict):
            for key2 in cfg[key1]:
                setattr(cfg, key2, cfg[key1][key2])


def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(
        osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir(imdb, weights_filename):
    """Return the directory where tensorflow summaries are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(
        osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            # else:
            #     raise ValueError(('Type mismatch ({} vs. {}) '
            #                       'for config key: {}').format(type(b[k]),
            #                                                    type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value