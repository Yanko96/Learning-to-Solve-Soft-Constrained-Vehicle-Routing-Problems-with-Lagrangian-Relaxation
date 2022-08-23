# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import collections
import json
import os
import random
import sys
import time
import six
import numpy as np
import copy
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from .parser import *

def np_to_tensor(inp, output_type, cuda_flag, volatile_flag=False):
	if output_type == 'float':
		inp_tensor = Variable(torch.FloatTensor(inp), volatile=volatile_flag)
	elif output_type == 'int':
		inp_tensor = Variable(torch.LongTensor(inp), volatile=volatile_flag)
	else:
		print('undefined tensor type')
	if cuda_flag:
		inp_tensor = inp_tensor.cuda()
	return inp_tensor

def load_dataset(filename, args):
	with open(filename, 'r') as f:
		samples = json.load(f)
	print('Number of data samples in ' + filename + ': ', len(samples))
	return samples


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