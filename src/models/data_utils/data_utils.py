import argparse
import collections
import copy
import json
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
import six
import torch
from torch.utils.data import DataLoader, Dataset

from .parser import *


def np_to_tensor(inp, output_type, cuda_flag):
    if output_type == 'float':
        inp_tensor = torch.FloatTensor(inp)
    elif output_type == 'int':
        inp_tensor = torch.LongTensor(inp)
    else:
        print('undefined tensor type')
    if cuda_flag:
        inp_tensor = inp_tensor.cuda()
    return inp_tensor

def load_dataset(filename):
    with open(filename, 'r') as f:
        samples = json.load(f)
    print('Number of data samples in ' + filename + ': ', len(samples))
    return samples

def collate_fn(batch):
    return batch

class vrpDataProcessor(object):
    def __init__(self):
        self.parser = vrpParser()

    def get_batch(self, data, batch_size, start_idx=None):
        data_size = len(data)
        if start_idx is not None:
            batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + batch_size))]
        else:
            batch_idxes = np.random.choice(len(data), batch_size)
        batch_data = []
        for idx in batch_idxes:
            problem = data[idx]
            dm = self.parser.parse(problem)
            batch_data.append(dm)
        return batch_data

class vrptwDataProcessor(object):
    def __init__(self):
        self.parser = vrptwParser()

    def get_batch(self, data, batch_size, start_idx=None):
        data_size = len(data)
        if start_idx is not None:
            batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + batch_size))]
        else:
            batch_idxes = np.random.choice(len(data), batch_size)
        batch_data = []
        for idx in batch_idxes:
            problem = data[idx]
            dm = self.parser.parse(problem)
            batch_data.append(dm)
        return batch_data

class vrptwDataset(Dataset):
    def __init__(self, data_path, pre_process=False):
        self.data_path = data_path
        self.samples = np.array(load_dataset(self.data_path))
        self.pre_process = pre_process
        if pre_process:
            pool = Pool()
            self.parser = vrptwParser()
            self.samples = np.array(pool.map(self.parser.parse, self.samples))
            # self.samples = np.array([self.parser.parse(sample) for sample in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.pre_process:
            return self.samples[idx]
        else:
            return self.parser.parse(self.samples[idx])
