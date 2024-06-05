# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import print_log, load_from_hdf5
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_datloader(args):
    """Generate dataloader"""
    dataset = mao_Dataset(args)
    args = dataset.args
    if args.train_rate + args.val_rate > 1:
        raise ValueError(f"train_rate + val_rate = {args.train_rate + args.val_rate} > 1 !")
    
    train_range = int(args.samples_per_class * args.train_rate)
    val_range = train_range + int(args.samples_per_class * args.val_rate) if args.val_rate > 0 else train_range

    train_list = [i for k in range(args.data_type) 
                    for j in range(k * args.typenum_samples, (k + 1) * args.typenum_samples, args.samples_per_class) 
                    for i in range(j, j + train_range)]
    test_list = [i for k in range(args.data_type) 
                   for j in range(k * args.typenum_samples, (k + 1) * args.typenum_samples, args.samples_per_class) 
                   for i in range(j + val_range, j + args.samples_per_class)]
    val_list = [i for k in range(args.data_type) 
                  for j in range(k * args.typenum_samples, (k + 1) * args.typenum_samples, args.samples_per_class) 
                  for i in range(j + train_range, j + val_range)] if args.val_rate > 0 else test_list


    train_dataset = Subset(dataset, train_list)
    test_dataset = Subset(dataset, test_list)
    valid_dataset = Subset(dataset, val_list) if args.val_rate > 0 else test_dataset
    

    print_log(f"Length of all dataset: {len(dataset)}")
    print_log(f"Length of train_dataset: {len(train_dataset)}")
    print_log(f"Length of valid_dataset: {len(valid_dataset)}")
    print_log(f"Length of test_dataset: {len(test_dataset)}")
    print_log(f"Shape of input_data: {test_dataset[0][0].shape}")


    # DataLoaders creation:
    if not args.dist:
        train_sampler = RandomSampler(train_dataset)
        vaild_sampler = SequentialSampler(valid_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        vaild_sampler = DistributedSampler(valid_dataset)

    train_loader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                num_workers=args.num_workers,
                                drop_last=args.drop_last,
                                pin_memory=True,
                                batch_size=args.per_device_train_batch_size)
    val_loader = DataLoader(valid_dataset,
                                sampler=vaild_sampler,
                                num_workers=args.num_workers,
                                drop_last=False,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size)
    test_loader = DataLoader(test_dataset,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size,
                                shuffle = False,
                                drop_last=False)
    return args, train_loader, val_loader, test_loader,


class mao_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the mao data '''
    def __init__(self, args):
        # Read inputs

        data = []
        for path in args.data_path:
            print_log(f"Loading {path}")
            data.append(load_from_hdf5(path,12))
            print_log(f"Successful load {path}")

        data = np.array(data)
        self.custom_length = args.total_samples

        data = data[..., args.select_channel, :]

        labels = np.repeat(np.arange(0, args.num_classes), args.samples_per_class)

        labels = np.tile(labels, args.data_type)

        traindata = data[:, :, :int(args.samples_per_class * args.train_rate)].reshape(-1, len(args.select_channel), args.sequence_length)

        args.data_mean = np.mean(traindata, axis=(0,2), keepdims=True)
        args.data_std = np.std(traindata, axis=(0,2), keepdims=True)

        data = data.reshape(-1, len(args.select_channel), args.sequence_length)
        self.inputs = torch.Tensor((data - args.data_mean) / args.data_std)
        self.labels = torch.LongTensor(labels)

        self.args = args


    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        return inputs, labels


    def __len__(self):
        # Returns the size of the dataset
        return self.custom_length
    

