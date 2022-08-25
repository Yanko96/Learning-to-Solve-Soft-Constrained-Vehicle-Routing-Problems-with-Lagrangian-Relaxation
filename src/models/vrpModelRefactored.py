import numpy as np
import operator
import random
import time
from multiprocessing.pool import ThreadPool

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence

from .data_utils import data_utils
from .modules import vrpInputEncoder, mlp
from .rewriter import vrpRewriter
from .BaseModel import BaseModel

eps = 1e-3
log_eps = np.log(eps)


class vrpModel(BaseModel):
    """
    Model architecture for vehicle routing.
    """
    def __init__(self, args):
        super(vrpModel, self).__init__(args)
        self.input_format = args.input_format
        self.embedding_size = args.embedding_size
        self.attention_size = args.attention_size
        self.sqrt_attention_size = int(np.sqrt(self.attention_size))
        self.reward_thres = -0.01
        self.input_encoder = vrpInputEncoder.SeqLSTM(args)
        self.policy_embedding = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 6 + self.embedding_size * 2, self.MLP_hidden_size, self.attention_size, self.cuda_flag, self.dropout_rate)
        self.policy = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 4, self.MLP_hidden_size, self.attention_size, self.cuda_flag, self.dropout_rate)
        self.value_estimator = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 4, self.MLP_hidden_size, 1, self.cuda_flag, self.dropout_rate)
        self.rewriter = vrpRewriter()

        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError('optimizer undefined: ', args.optimizer)


    def rewrite(self, dm, trace_rec, padded_predicted_rewards, sample_rewrite_pos, eval_flag, max_search_pos, reward_thres=None):
        candidate_dm = []
        candidate_rewrite_rec = []
        candidate_trace_rec = []
        candidate_scores = []

        for idx, rewrite_pos in enumerate(sample_rewrite_pos):
            pred_reward = padded_predicted_rewards[rewrite_pos]
            rewrite_pos += 1
            if len(candidate_dm) > 0 and idx >= max_search_pos:
                break
            if reward_thres is not None and pred_reward < reward_thres:
                if eval_flag:
                    break
                elif np.random.random() > 0.1:
                    continue
            candidate_neighbor_idxes = dm.get_neighbor_idxes(rewrite_pos)
            cur_node_idx = dm.vehicle_state[rewrite_pos][0]
            cur_node = dm.get_node(cur_node_idx)
            pre_node_idx = dm.vehicle_state[rewrite_pos - 1][0]
            pre_node = dm.get_node(pre_node_idx)
            pre_capacity = dm.vehicle_state[rewrite_pos - 1][1]
            depot = dm.get_node(0)
            depot_state = dm.encoder_outputs[0].unsqueeze(0)
            cur_state = dm.encoder_outputs[rewrite_pos].unsqueeze(0)
            cur_states_0 = []
            cur_states_1 = []
            cur_states_2 = []
            new_embeddings_0 = []
            new_embeddings_1 = []
            for i in candidate_neighbor_idxes:
                neighbor_idx = dm.vehicle_state[i][0]
                neighbor_node = dm.get_node(neighbor_idx)
                cur_states_0.append(depot_state.clone())
                cur_states_1.append(cur_state.clone())
                cur_states_2.append(dm.encoder_outputs[i].unsqueeze(0))
                if pre_capacity >= neighbor_node.demand:
                    new_embedding = [neighbor_node.x, neighbor_node.y, neighbor_node.demand * 1.0 / dm.capacity, pre_node.x, pre_node.y, neighbor_node.demand * 1.0 / pre_capacity, dm.get_dis(pre_node, neighbor_node)]
                else:
                    new_embedding = [neighbor_node.x, neighbor_node.y, neighbor_node.demand * 1.0 / dm.capacity, pre_node.x, pre_node.y, neighbor_node.demand * 1.0 / dm.capacity, dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]
                new_embeddings_0.append(new_embedding[:])
                if pre_capacity >= neighbor_node.demand:
                    new_embedding = [(neighbor_node.x - depot.x) * (pre_node.x - depot.x), (neighbor_node.y - depot.y) * (pre_node.y - depot.y), (neighbor_node.demand - cur_node.demand) * 1.0 / pre_capacity, pre_node.px, pre_node.py, \
                    (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity, dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]
                else:
                    new_embedding = [(neighbor_node.x - depot.x) * (pre_node.x - depot.x), (neighbor_node.y - depot.y) * (pre_node.y - depot.y), (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity, pre_node.px, pre_node.py, \
                    (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity, dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]					
                new_embeddings_1.append(new_embedding[:])
            cur_states_0 = torch.cat(cur_states_0, 0)
            cur_states_1 = torch.cat(cur_states_1, 0)
            cur_states_2 = torch.cat(cur_states_2, 0)
            new_embeddings_0 = data_utils.np_to_tensor(new_embeddings_0, 'float', False)
            new_embeddings_1 = data_utils.np_to_tensor(new_embeddings_1, 'float', False)
            policy_inputs = torch.cat([cur_states_0, cur_states_1, cur_states_2, new_embeddings_0, new_embeddings_1], 1)
            ctx_embeddings = self.policy_embedding(policy_inputs)
            cur_state_key = self.policy(torch.cat([cur_state, depot_state], dim=1))
            ac_logits = torch.matmul(cur_state_key, torch.transpose(ctx_embeddings, 0, 1)) / 64
            ac_logprobs = F.log_softmax(ac_logits, dim=1)
            ac_probs = F.softmax(ac_logits, dim=1)
            ac_logits = ac_logits.squeeze(0)
            ac_logprobs = ac_logprobs.squeeze(0)
            ac_probs = ac_probs.squeeze(0)
            if eval_flag:
                _, candidate_acs = torch.sort(ac_logprobs, descending=True)
                candidate_acs = candidate_acs.data.cpu().numpy()
            else:
                candidate_acs_dist = Categorical(ac_probs)
                candidate_acs = candidate_acs_dist.sample(sample_shape=[ac_probs.size()[0]])
                #candidate_acs = torch.multinomial(ac_probs, ac_probs.size()[0])
                candidate_acs = candidate_acs.data.cpu().numpy()
                indexes = np.unique(candidate_acs, return_index=True)[1]
                candidate_acs = [candidate_acs[i] for i in sorted(indexes)]

            for i in candidate_acs:
                neighbor_idx = candidate_neighbor_idxes[i]
                new_dm = self.rewriter.move(dm, rewrite_pos, neighbor_idx)
                if new_dm.tot_dis[-1] in trace_rec:
                    continue
                candidate_dm.append(new_dm)
                candidate_rewrite_rec.append((ac_logprobs, pred_reward, rewrite_pos, i, new_dm.tot_dis[-1]))
                if len(candidate_dm) >= max_search_pos:
                    break
        return candidate_dm, candidate_rewrite_rec


    def batch_rewrite(self, dm_list, trace_rec, pred_rewards, eval_flag, max_search_pos, reward_thres):
        candidate_dm = []
        candidate_rewrite_rec = []
        lengths = [len(dm.vehicle_state)-2 for dm in dm_list]
        
        start_idx = 0
        padded_predicted_rewards = []
        for length in lengths:
            padded_predicted_rewards.append(pred_rewards[start_idx:start_idx+length])
            start_idx += length
        padded_predicted_rewards = pad_sequence(padded_predicted_rewards, batch_first=True, padding_value=-float('inf')).squeeze()
        exp_padded_predicted_rewards = torch.exp(padded_predicted_rewards * 10)
        batch_rewrite_pos_dist = Categorical(exp_padded_predicted_rewards.squeeze())
        batch_rewrite_pos = batch_rewrite_pos_dist.sample(sample_shape=[max(lengths)])
        batch_rewrite_pos = batch_rewrite_pos.permute(1, 0)
            
        for i in range(len(dm_list)):
            sample_rewrite_pos = torch.unique(batch_rewrite_pos[i], sorted=False).flip(dims=[0])
            cur_candidate_dm, cur_candidate_rewrite_rec = self.rewrite(dm_list[i], trace_rec[i], padded_predicted_rewards[i], sample_rewrite_pos, eval_flag, max_search_pos, reward_thres)
            candidate_dm.append(cur_candidate_dm)
            candidate_rewrite_rec.append(cur_candidate_rewrite_rec)
        return candidate_dm, candidate_rewrite_rec


    def forward(self, batch_data, eval_flag=False):
        eval_flag = False
        torch.set_grad_enabled(not eval_flag)
        batch_size = len(batch_data)
        dm_list, encoder_output = self.input_encoder.calc_embedding(batch_data, eval_flag)

        active = True
        reduce_steps = 0

        trace_rec = [{} for _ in range(batch_size)]
        rewrite_rec = [[] for _ in range(batch_size)]
        dm_rec = [[] for _ in range(batch_size)]

        for idx in range(batch_size):
            dm_rec[idx].append(dm_list[idx])
            trace_rec[idx][dm_list[idx].tot_dis[-1]] = 0

        while active and (self.max_reduce_steps is None or reduce_steps < self.max_reduce_steps):
            active = False
            reduce_steps += 1
            node_idx = [i for dm in dm_list for i in range(1, len(dm.vehicle_state) - 1)]
            depot_idx = [0 for dm in dm_list for i in range(1, len(dm.vehicle_state) - 1)]
            sample_idx = [idx for idx, dm in enumerate(dm_list) for state in dm.vehicle_state[1:-1]]
            node_states = encoder_output[sample_idx, node_idx]
            depot_states = encoder_output[sample_idx, depot_idx]

            pred_rewards = self.value_estimator(torch.cat([node_states, depot_states], dim=1))
            candidate_dm, candidate_rewrite_rec = self.batch_rewrite(dm_list, trace_rec, pred_rewards, eval_flag, max_search_pos=1, reward_thres=self.reward_thres)

            for dm_idx in range(batch_size):
                cur_candidate_dm = candidate_dm[dm_idx]
                cur_candidate_rewrite_rec = candidate_rewrite_rec[dm_idx]
                if len(cur_candidate_dm) > 0:
                    active = True
                    cur_dm = cur_candidate_dm[0]
                    cur_rewrite_rec = cur_candidate_rewrite_rec[0]
                    dm_list[dm_idx] = cur_dm
                    rewrite_rec[dm_idx].append(cur_rewrite_rec)
                    trace_rec[dm_idx][cur_dm.tot_dis[-1]] = 0
            if not active:
                break

            updated_dm, encoder_output = self.input_encoder.calc_embedding(dm_list, eval_flag)
            for i in range(batch_size):
                if updated_dm[i].tot_dis[-1] != dm_rec[i][-1].tot_dis[-1]:
                    dm_rec[i].append(updated_dm[i])

        total_policy_loss = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)
        total_value_loss = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)

        pred_value_rec = []
        value_target_rec = []
        total_reward = 0
        total_rewrite_steps = 0
        for dm_idx, cur_dm_rec in enumerate(dm_rec):
            pred_dis = []
            for dm in cur_dm_rec:
                pred_dis.append(dm.tot_dis[-1])
            best_reward = pred_dis[0]

            for idx, (ac_logprob, pred_reward, rewrite_pos, applied_op, new_dis) in enumerate(rewrite_rec[dm_idx]):
                cur_reward = pred_dis[idx] - pred_dis[idx + 1]
                best_reward = min(best_reward, pred_dis[idx + 1])

                if self.gamma > 0.0:
                    decay_coef = 1.0
                    num_rollout_steps = len(cur_dm_rec) - idx - 1
                    for i in range(idx + 1, idx + 1 + num_rollout_steps):
                        cur_reward = max(decay_coef * (pred_dis[idx] - pred_dis[i]), cur_reward)
                        decay_coef *= self.gamma

                cur_reward_tensor = data_utils.np_to_tensor(np.array([cur_reward], dtype=np.float32), 'float', self.cuda_flag)
                if ac_logprob.data.cpu().numpy()[0] > log_eps or cur_reward - pred_reward > 0:
                    ac_mask = np.zeros(ac_logprob.size()[0])
                    ac_mask[applied_op] = cur_reward - pred_reward
                    ac_mask = data_utils.np_to_tensor(ac_mask, 'float', self.cuda_flag)
                    total_policy_loss -= ac_logprob[applied_op] * ac_mask[applied_op]
                pred_value_rec.append(pred_reward)
                value_target_rec.append(cur_reward_tensor)
            
            total_reward += best_reward

        if len(pred_value_rec) > 0:
            pred_value_rec = torch.stack(pred_value_rec, 0)
            value_target_rec = torch.cat(value_target_rec, 0)
            pred_value_rec = pred_value_rec.unsqueeze(1)
            value_target_rec = value_target_rec.unsqueeze(1)
            total_value_loss = F.smooth_l1_loss(pred_value_rec, value_target_rec, reduction='sum')
        total_policy_loss /= batch_size
        total_value_loss /= batch_size
        total_loss = total_policy_loss * self.value_loss_coef + total_value_loss
        total_reward = total_reward * 1.0 / batch_size

        return total_loss, total_reward, dm_rec