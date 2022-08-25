import argparse
import json
import os
import re
from statistics import mean
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm

from ..data_utils import data_utils
from ..data_utils.parser import *

CKPT_PATTERN = re.compile('^ckpt-(\d+)$')


class Supervisor(object):
    """
    The base class to manage the high-level model execution processes. The concrete classes for different applications are derived from it.
    """
    def __init__(self, model, args):
        self.processes = args.processes
        self.model = model
        self.keep_last_n = args.keep_last_n
        self.dropout_rate = args.dropout_rate
        self.global_step = args.resume
        self.batch_size = args.batch_size
        self.model_dir = args.model_dir


    def load_pretrained(self, load_model):
        print("Read model parameters from %s." % load_model)
        checkpoint = torch.load(load_model)
        self.model.load_state_dict(checkpoint)


    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        global_step_padded = format(self.global_step, '08d')
        ckpt_name = 'ckpt-' + global_step_padded
        path = os.path.join(self.model_dir, ckpt_name)
        ckpt = self.model.state_dict()
        torch.save(ckpt, path)

        if self.keep_last_n is not None:
            ckpts = []
            for file_name in os.listdir(self.model_dir):
                matched_name = CKPT_PATTERN.match(file_name)
                if matched_name is None or matched_name == ckpt_name:
                    continue
                step = int(matched_name.group(1))
                ckpts.append((step, file_name))
            if len(ckpts) > self.keep_last_n:
                ckpts.sort()
                os.unlink(os.path.join(self.model_dir, ckpts[0][1]))


class vrpSupervisor(Supervisor):
    """
    Management class for vehicle routing.
    """
    def __init__(self, model, args):
        super(vrpSupervisor, self).__init__(model, args)
        self.DataProcessor = data_utils.vrpDataProcessor()


    def train(self, batch_data):
        self.model.dropout_rate = self.dropout_rate
        self.model.optimizer.zero_grad()
        avg_loss, avg_reward, dm_rec = self.model(batch_data)
        self.global_step += 1
        if type(avg_loss) != float:
            avg_loss.backward()
            self.model.train()
        return avg_loss.item(), avg_reward


    def batch_eval(self, eval_data, output_trace_flag, process_idx):
        cum_loss = 0
        cum_reward = 0
        data_size = len(eval_data)

        for batch_idx in range(0, data_size, self.batch_size):
            batch_data = self.DataProcessor.get_batch(eval_data, self.batch_size, batch_idx)
            cur_avg_loss, cur_avg_reward, dm_rec = self.model(batch_data, eval_flag=True)
            cum_loss += cur_avg_loss.item() * len(batch_data)
            cum_reward += cur_avg_reward * len(batch_data)
            if output_trace_flag == 'complete':
                for cur_dm_rec in dm_rec:
                    for i in range(len(cur_dm_rec)):
                        print('step ' + str(i))
                        dm = cur_dm_rec[i]
                        print(dm.tot_dis[-1])
                        for j in range(len(dm.vehicle_state)):
                            cur_pos, cur_capacity = dm.vehicle_state[j]
                            cur_node = dm.get_node(cur_pos)
                            print(cur_node.x, cur_node.y, cur_node.demand, cur_capacity, dm.tot_dis[j])
                        print('')
            print('process start idx: %d batch idx: %d pred reward: %.4f' \
                % (process_idx, batch_idx, cur_avg_reward))
        return cum_loss, cum_reward


    def eval(self, data, output_trace_flag, max_eval_size=None):
        data_size = len(data)
        if max_eval_size is not None:
            data_size = min(data_size, max_eval_size)
        eval_data = data[:data_size]
        if self.processes == 1:
            cum_loss, cum_reward = self.batch_eval(eval_data, output_trace_flag, 0)
        else:
            cum_loss = 0
            cum_reward = 0
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            pool = mp.Pool(processes=self.processes)
            res = []
            batch_per_process = data_size // self.processes
            if data_size % batch_per_process > 0:
                batch_per_process += 1
            for st in range(0, data_size, batch_per_process):
                res += [pool.apply_async(self.batch_eval, (eval_data[st: st + batch_per_process], output_trace_flag, st))]
            for i in range(len(res)):
                cur_cum_loss, cur_cum_reward = res[i].get()
                cum_loss += cur_cum_loss
                cum_reward += cur_cum_reward

        avg_loss = cum_loss / data_size
        avg_reward = cum_reward / data_size
        print('average pred reward: %.4f' % avg_reward)
        return avg_loss, avg_reward

class vrptwSupervisor(Supervisor):
    """
    Management class for vehicle routing.
    """
    def __init__(self, model, args, dataloader):
        super(vrptwSupervisor, self).__init__(model, args)
        self.dataloader = dataloader


    def train(self, batch_data):
        self.model.dropout_rate = self.dropout_rate
        self.model.optimizer.zero_grad()
        avg_loss, avg_reward, dm_rec = self.model(batch_data)
        self.global_step += 1
        if type(avg_loss) != float:
            avg_loss.backward()
            self.model.train()
        avg_dist = np.mean([np.min([dm.tot_dis[-1] for dm in dm_rec[i]]) for i in range(len(dm_rec))])
        return avg_loss.item(), avg_reward, avg_dist


    def batch_eval(self, eval_data, output_trace_flag, process_idx):
        cum_loss = 0
        cum_reward = 0
        data_size = len(eval_data)

        for batch_idx in range(0, data_size, self.batch_size):
            batch_data = self.DataProcessor.get_batch(eval_data, self.batch_size, batch_idx)
            cur_avg_loss, cur_avg_reward, dm_rec = self.model(batch_data, eval_flag=True)
            cum_loss += cur_avg_loss.item() * len(batch_data)
            cum_reward += cur_avg_reward * len(batch_data)
            if output_trace_flag == 'complete':
                for cur_dm_rec in dm_rec:
                    for i in range(len(cur_dm_rec)):
                        print('step ' + str(i))
                        dm = cur_dm_rec[i]
                        print(dm.tot_dis[-1])
                        for j in range(len(dm.vehicle_state)):
                            cur_pos, cur_capacity = dm.vehicle_state[j]
                            cur_node = dm.get_node(cur_pos)
                            print(cur_node.x, cur_node.y, cur_node.demand, cur_capacity, dm.tot_dis[j])
                        print('')
            print('process start idx: %d batch idx: %d pred reward: %.4f' \
                % (process_idx, batch_idx, cur_avg_reward))
        return cum_loss, cum_reward


    def eval(self, data, output_trace_flag, max_eval_size=None):
        data_size = len(data)
        if max_eval_size is not None:
            data_size = min(data_size, max_eval_size)
        eval_data = data[:data_size]
        if self.processes == 1:
            cum_loss, cum_reward = self.batch_eval(eval_data, output_trace_flag, 0)
        else:
            cum_loss = 0
            cum_reward = 0
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            pool = mp.Pool(processes=self.processes)
            res = []
            batch_per_process = data_size // self.processes
            if data_size % batch_per_process > 0:
                batch_per_process += 1
            for st in range(0, data_size, batch_per_process):
                res += [pool.apply_async(self.batch_eval, (eval_data[st: st + batch_per_process], output_trace_flag, st))]
            for i in range(len(res)):
                cur_cum_loss, cur_cum_reward = res[i].get()
                cum_loss += cur_cum_loss
                cum_reward += cur_cum_reward

        avg_loss = cum_loss / data_size
        avg_reward = cum_reward / data_size
        print('average pred reward: %.4f' % avg_reward)
        return avg_loss, avg_reward
