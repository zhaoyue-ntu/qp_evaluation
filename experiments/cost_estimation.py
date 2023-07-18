import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('../evaluation/')

from dataset_utils import *

dat_path = '../data/stats/'
dat_dict = get_stats(dat_path)

# dat_path = '../data/tpch/'
# dat_dict = get_tpch(dat_path)

# dat_path = '../data/tpcds/'
# dat_dict = get_tpcds(dat_path)

# dat_path = '../data/imdb/'
# dat_dict = get_imdb(dat_path)


ds_info = dat_dict['ds_info']
train_roots = dat_dict['train_roots']
train_costs = dat_dict['train_costs']

val_roots = dat_dict['val_roots']
val_costs = dat_dict['val_costs']

test_roots = dat_dict['test_roots']
test_costs = dat_dict['test_costs']
#imdb
# syn_roots = dat_dict['syn_roots']
# syn_costs = dat_dict['syn_costs']
# job_light_roots = dat_dict['job_light_roots']
# job_light_costs = dat_dict['job_light_costs']


# Method specific
from algorithms.avgdl import *
encoding = Encoding()

class Args:
    device = 'cuda:0'
    bs = 1024
    epochs = 401
    lr = 1e-3
    save_path = 'results/cost/avgdl/stats/'
args = Args()

import os
save_path = args.save_path 
if not os.path.exists(save_path):
    os.makedirs(save_path)

model = AVGDL(32, 64, 64)
device = args.device

ds = AVGDL_Dataset(train_roots, encoding, train_costs, ds_info)
train_loader = DataLoader(dataset=ds,
                          batch_size = args.bs,
                          collate_fn=collate,
                          shuffle=True)

val_ds = AVGDL_Dataset(val_roots, encoding, val_costs, ds_info)
val_loader = DataLoader(dataset=val_ds,
                          batch_size = args.bs,
                          collate_fn=collate,
                          shuffle=False)

test_ds = AVGDL_Dataset(test_roots, encoding, test_costs, ds_info)
test_loader = DataLoader(dataset=test_ds,
                          batch_size = args.bs,
                          collate_fn=collate,
                          shuffle=False)

from trainer import *
prediction = Prediction(64)
# connect representation method to prediction model
model_comb = nn.Sequential(model, prediction)
train(model_comb, train_loader, val_loader, val_costs, ds_info, args)

evaluate(model_comb, test_loader, test_costs, 1, ds_info.cost_norm, args.device)