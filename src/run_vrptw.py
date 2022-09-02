import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import arguments
import models.data_utils.data_utils as data_utils
import models.model_utils as model_utils
from models.vrptwModel import vrptwModel


def create_model(args):
    model = vrptwModel(args)

    if model.cuda_flag:
        model = model.cuda()
    # model.share_memory()
    model_supervisor = model_utils.vrptwSupervisor(model, args)
    if args.load_model:
        model_supervisor.load_pretrained(args.load_model)
    elif args.resume:
        pretrained = 'ckpt-' + str(args.resume).zfill(8)
        print('Resume from {} iterations.'.format(args.resume))
        model_supervisor.load_pretrained(args.model_dir+'/'+pretrained)
    else:
        print('Created model with fresh parameters.')
        model_supervisor.model.init_weights(args.param_init)
    return model_supervisor


def train(args):
    print('Training:')
    train_data = data_utils.vrptwDataset(args.train_dataset, True)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=data_utils.collate_fn)

    train_data_size = len(train_data)

    eval_data = data_utils.vrptwDataset(args.val_dataset, True)
    eval_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=data_utils.collate_fn)

    model_supervisor = create_model(args)

    resume_idx = args.resume * args.batch_size

    running_mean_dist = 0.0
    running_mean_cost = 0.0
    running_mean_penalty = 0.0
    logger = model_utils.Logger(args)
    if args.resume:
        logs = pd.read_csv(os.path.join(args.model_dir + args.log_name))
        for index, log in logs.iterrows():
            val_summary = {'eval_avg_cost': log['eval_avg_cost'], 'eval_avg_dist': log['eval_avg_dist'], 'eval_avg_penalty': log['eval_avg_penalty'], 'global_step': log['global_step']}
            logger.write_summary(val_summary)

    for epoch in range(resume_idx//train_data_size, args.num_epochs):
        for batch_idx, samples in enumerate(train_dataloader):
            print("Epoch {}, Batch {}".format(epoch, batch_idx))
            train_loss, (avg_cost, avg_dist, avg_penalty) = model_supervisor.train(samples)
            running_mean_dist = running_mean_dist * 0.95 + avg_dist * 0.05
            running_mean_cost = running_mean_cost * 0.95 + avg_cost * 0.05
            running_mean_penalty = running_mean_penalty * 0.95 + avg_penalty * 0.05
            print('train loss: %.3f avg cost: %.3f avg distance: %.3f avg penalty: %.3f run avg cost: %.3f run avg distance: %.3f run avg penalty: %.3f'% (train_loss, avg_cost, avg_dist, avg_penalty, running_mean_cost, running_mean_dist, running_mean_penalty))

            if model_supervisor.global_step % args.eval_every_n == 0:
                eval_loss, (eval_avg_cost, eval_avg_dist, eval_avg_penalty) = model_supervisor.eval(eval_dataloader, args.output_trace_flag, args.max_eval_size)
                val_summary = {'eval_avg_cost': eval_avg_cost, 'eval_avg_dist': eval_avg_dist, 'eval_avg_penalty': eval_avg_penalty, 'global_step': model_supervisor.global_step}
                logger.write_summary(val_summary)
                model_supervisor.save_model()

            if args.lr_decay_steps and model_supervisor.global_step % args.lr_decay_steps == 0:
                model_supervisor.model.lr_decay(args.lr_decay_rate)
                if model_supervisor.model.cont_prob > 0.01:
                    model_supervisor.model.cont_prob *= 0.5


def evaluate(args):
    print('Evaluation:')

    test_data = data_utils.vrptwDataset(args.test_dataset, True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=data_utils.collate_fn)
    test_data_size = len(test_data)
    args.dropout_rate = 0.0

    model_supervisor = create_model(args)
    test_loss, (test_avg_cost, test_avg_dist, test_avg_penalty) = model_supervisor.eval(test_dataloader, args.output_trace_flag)


    print('test loss: %.4f test cost: %.4f test distance: %.4f test penalty: %.4f' % (test_loss, test_avg_cost, test_avg_dist, test_avg_penalty))


if __name__ == "__main__":
    argParser = arguments.get_arg_parser("vrptw")
    args = argParser.parse_args()
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.eval:
        evaluate(args)
    else:
        train(args)
