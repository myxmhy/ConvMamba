# -*- coding: utf-8 -*-
import torch.distributed as dist
from collections import OrderedDict
import json
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torch
from typing import Tuple
import os
import logging
import h5py
import concurrent.futures

def print_rank_0(message):
    """Only output in root process or single process
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def print_log(message):
    print_rank_0(message)
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logging.info(message)
    else:
        logging.info(message)

def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def save_json(data,data_path):
    """Save json data
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            with open(data_path, 'w') as json_file:
                json.dump(data,json_file,indent=4)
    else:
        with open(data_path, 'w') as json_file:
            json.dump(data,json_file,indent=4)


def json2Parser(json_path):
    """Load json and return a parser-like object
    Parameters
    ----------
    json_path : str
        The json file path.
    
    Returns
    -------
    args : easydict.EasyDict
        A parser-like object.
    """
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)


def reduce_tensor(tensor):
    rt = tensor.data.clone()
    dist.all_reduce(rt.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return rt


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    if deterministic:
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    ban_list = ["setup_config", "data_config", "optim_config", "sched_config", "model_config", "ds_config", ]
    for k, v in configs.items():
        if k not in ban_list:
            message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def save_to_hdf5(data, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('dataset', data=data)


def load_partial_from_hdf5(filename, start, end):
    with h5py.File(filename, 'r') as f:
        data = f['dataset'][start:end]
    return data

def load_from_hdf5(filename, parts=8):
    with h5py.File(filename, 'r') as f:
        dataset_size = f['dataset'].shape[0]
        chunk_size = dataset_size // parts
        data_shape = f['dataset'].shape

    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(parts)]
    ranges[-1] = (ranges[-1][0], dataset_size)

    data = np.zeros(data_shape, dtype=np.float64)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(load_partial_from_hdf5, filename, start, end): idx for idx, (start, end) in enumerate(ranges)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                (start, end) = ranges[idx]
                data[start:end] = result
            except Exception as e:
                print(f"Failed to load part of the file: {e}")
    
    return data


