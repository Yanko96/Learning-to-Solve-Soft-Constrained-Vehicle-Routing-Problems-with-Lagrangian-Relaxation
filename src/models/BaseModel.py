import operator
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm


class BaseModel(nn.Module):
	"""
	Base neural rewriter model. The concrete architectures for different applications are derived from it.
	"""
	def __init__(self, args):
		super(BaseModel, self).__init__()
		self.processes = args.processes
		self.batch_size = args.batch_size
		self.LSTM_hidden_size = args.LSTM_hidden_size
		self.MLP_hidden_size = args.MLP_hidden_size
		self.num_MLP_layers = args.num_MLP_layers
		self.gradient_clip = args.gradient_clip
		if args.lr_decay_steps and args.resume:
			self.lr = args.lr * args.lr_decay_rate ** ((args.resume - 1) // args.lr_decay_steps)
		else:
			self.lr = args.lr
		print('Current learning rate is {}.'.format(self.lr))
		self.dropout_rate = args.dropout_rate
		self.max_reduce_steps = args.max_reduce_steps
		self.num_sample_rewrite_pos = args.num_sample_rewrite_pos
		self.num_sample_rewrite_op = args.num_sample_rewrite_op
		self.value_loss_coef = args.value_loss_coef
		self.gamma = args.gamma
		self.cont_prob = args.cont_prob
		self.cuda_flag = args.cuda


	def init_weights(self, param_init):
		for param in self.parameters():
			param.data.uniform_(-param_init, param_init)


	def lr_decay(self, lr_decay_rate):
		self.lr *= lr_decay_rate
		print('Current learning rate is {}.'.format(self.lr))
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr


	def train(self):
		if self.gradient_clip > 0:
			clip_grad_norm(self.parameters(), self.gradient_clip)
		self.optimizer.step()
