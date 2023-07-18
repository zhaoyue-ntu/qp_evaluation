import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import os
from sklearn.metrics import f1_score
import torch.nn.functional as F


##evaluation functions according to AIMEETSAI paper

# collate labels
# CLIP_VAL = 1e4
CUT_OFF = 0.2
IMPROVE = 1
REGRESS = 0
NODIFF = 2

def collate_labels(df):
    # label the plan pairs by: IMPROVE(1), REGRESS(0), and NODIFF(2)
    labels = []
    pair_diffs = []
    pair_diff_ratios = []
    for idx, row in df.iterrows():

        relative_time = row['Right_Cost'] / row['Left_Cost']
        if relative_time > (1+CUT_OFF):
            labels.append(REGRESS)
        elif relative_time < (1-CUT_OFF):
            labels.append(IMPROVE)
        else: 
            labels.append(NODIFF)
    
    return labels


def compute_score(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    acc = sum(pred == gt) / len(gt)
    f1 = f1_score(gt, pred, average = None)
    avg_f1 = np.mean(f1)
    return acc, f1, avg_f1


# helper for splitting
def hash_by_plan(plan):
    # returns the hash value of a plan's node types
    node_types = []
    def dfs(node):
        if 'Plans' in node:
            for child in node['Plans']:
                dfs(child)
        elif 'Plan' in node:
            # imdb
            if 'Plans' in node['Plan']:
                for child in node['Plan']['Plans']:
                    dfs(child)
        node_types.append(node['Node Type'])
    dfs(plan)
    return hash(tuple(node_types))

def add_plan_id(raw_df):
    # append an id to each unique plan hash value
    hash2id = {}
    ids = []
    for i, row in raw_df.iterrows():
        if 'id' in row:
            # imdb
            idx = row['id']
            js_str = row['json']
        else:
            idx = i
            js_str = row['Plan_dump']
            
        if js_str == 'failed':
            continue   
        plan = json.loads(js_str)
        has = hash_by_plan(plan)
        if has not in hash2id:
            hash2id[has] = len(hash2id)
        ids.append(hash2id[has])
    raw_df['Plan_id'] = ids


def split_grouped_ids(raw_df, threshold):
    # group dataset by their plan ids, and split into train ids and test ids by given threshold
    add_plan_id(raw_df)
    #### need to do with imdb's query id
    template2plan_id = dict(raw_df.groupby('Query_id')['Plan_id'].unique())
    data_raw_train_ids = set()
    data_raw_test_ids = set()
    for k, v in template2plan_id.items():
        if len(v) <= threshold:
            cur = set(raw_df.loc[(raw_df['Plan_id'].isin(v)) & (raw_df['Query_id']==k)].index)
            data_raw_train_ids.update(cur)
        else:
            tv = v[:len(v)*(threshold-1)//threshold]
            vv = v[len(v)*(threshold-1)//threshold:]
            cur = set(raw_df.loc[(raw_df['Plan_id'].isin(tv)) & (raw_df['Query_id']==k)].index)
            data_raw_train_ids.update(cur)
            cur = set(raw_df.loc[(raw_df['Plan_id'].isin(vv)) & (raw_df['Query_id']==k)].index)
            data_raw_test_ids.update(cur)    
    return data_raw_train_ids, data_raw_test_ids

def split_train_test(df, raw_df, method = 'pair', threshold = 5, group = 0, dataset = None):
    np.random.seed(42)
    length = len(df)
    if method == 'pair':
        order = np.random.permutation(length)
        train_idxs = order[:length*(threshold-1)//threshold]
        test_idxs = order[length*(threshold-1)//threshold:]
        train_pairs = df.loc[train_idxs]
        test_pairs = df.loc[test_idxs]
    
    elif method == 'query':
        #### need to do with imdb's run id
        max_run_id = max(df['Query_id'])
        order = np.random.permutation(max_run_id+1)
        train_run_id = order[:max_run_id*(threshold-1)//threshold]
        test_run_id = order[max_run_id*(threshold-1)//threshold:]
        train_pairs = df.loc[df['Query_id'].isin(train_run_id)]
        test_pairs = df.loc[df['Query_id'].isin(test_run_id)]
    
    elif method == 'plan':
        data_raw_train_ids, data_raw_test_ids = split_grouped_ids(raw_df, threshold)
        train_pairs = df.loc[
            (df['Left'].isin(data_raw_train_ids)) & 
            df['Right'].isin(data_raw_train_ids)
        ]
        test_pairs = df.loc[
            (df['Right'].isin(data_raw_test_ids)) |
            (df['Left'].isin(data_raw_test_ids))           
        ]
        
    ## for TPC-H and TPC-DS, so that queries are indeed different
    ## different templates have different complexity, especially in TPC-H
    ## thus average from a few groups
    ## TPC-H: 0,2,6
    ## TPC-DS: 0,1,2,4,5
    elif method == 'template': 
        if dataset == 'TPCH':
            group_size = 2
        elif dataset == 'TPCDS':
            group_size = 11
        
        templates = df['Query_id'].unique()
        if isinstance(group, list):
            train_pairs = df.loc[~df['Query_id'].isin(group)]
            test_pairs = df.loc[df['Query_id'].isin(group)]           
        else: 
            group_id = list(range(group*group_size, (group+1)*group_size))
            template_id = templates[group_id]
            train_pairs = df.loc[~df['Query_id'].isin(template_id)]
            test_pairs = df.loc[df['Query_id'].isin(template_id)]
    
    return train_pairs, test_pairs

def randomSwap(df):
    ddf = df.copy()
    length = len(df)
    np.random.seed(42)
    
    truth_val = np.random.randint(2, size=length)
    to_swap = [i for i, t in enumerate(truth_val) if t]
    
    print(to_swap[:10])
    ddf.loc[to_swap, 'Left'] = df.loc[to_swap, 'Right']
    ddf.loc[to_swap, 'Right'] = df.loc[to_swap, 'Left']
    
    ddf.loc[to_swap, 'Left_Cost'] = df.loc[to_swap, 'Right_Cost']
    ddf.loc[to_swap, 'Right_Cost'] = df.loc[to_swap, 'Left_Cost']
    
    return ddf

import sys
sys.path.append('../evaluation/')

from algorithms.avgdl import AVGDL_Dataset, AVGDL
from algorithms.avgdl import Encoding as avgdl_Encoding
from algorithms.avgdl import DataLoader as avgdl_loader
from algorithms.avgdl import collate as avgdl_collate

def Loader(left_ds, right_ds, args):
 
    if args.method == 'avgdl':
        left_ld = avgdl_loader(left_ds, batch_size = args.bs, collate_fn=avgdl_collate, shuffle=False)
        left_batch = next(iter(left_ld))[0].to(args.device)
        
        right_ld = avgdl_loader(right_ds, batch_size = args.bs, collate_fn=avgdl_collate, shuffle=False)
        right_batch = next(iter(right_ld))[0].to(args.device)

    
    return left_batch, right_batch

def get_dat_model(roots, costs, args):
    if args.method == 'avgdl':    
        encoding = avgdl_Encoding()
        full_ds = AVGDL_Dataset(roots, encoding, costs, ds_info)
        rep = AVGDL(32, 64, 64)
        model = Classifier(64)
        
    return full_ds, rep, model

class Classifier(nn.Module):
    def __init__(self, in_feat, hid_unit=64, classes=3):
        super(Classifier, self).__init__()
        self.mlp1 = nn.Linear(in_feat, hid_unit)
        self.mlp2 = nn.Linear(hid_unit, hid_unit)
        self.mlp3 = nn.Linear(hid_unit, hid_unit)
        self.mlp4 = nn.Linear(hid_unit, classes)
    def forward(self, lefts, rights):
        features = rights - lefts
        hid = F.relu(self.mlp1(features))
        mid = F.relu(self.mlp2(hid))
        mid = F.relu(self.mlp3(mid))
        out = self.mlp4(hid+mid)
        return out
    
    
# training functions

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

import time
def train(model, rep, full_ds, train_df, ds_info, args):
    
    bs, device, epochs = args.bs, args.device, args.epochs
    lr = args.lr
    
    if rep == 'NA':
        optimizer = torch.optim.Adam(list(model.parameters()),lr = args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters())+ list(rep.parameters()),lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.8)
    crit = nn.CrossEntropyLoss()
    best_acc = 0
    rng = np.random.default_rng()

#     t0 = time.time()
    best_prev_f1 = 0

    model = model.to(device)
    if rep != 'NA':
        rep = rep.to(device)
    
    
#     best_model_path = None
    t0 = time.time()
    for epoch in range(epochs):
        losses = 0
        model.train()
        predlables = []
        gt = []

        train_idxs = rng.permutation(len(train_df))
        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()

            lefts = train_df.loc[idxs, 'Left'].to_numpy()
            rights = train_df.loc[idxs, 'Right'].to_numpy()

            left_ds = torch.utils.data.Subset(full_ds, lefts)
            right_ds = torch.utils.data.Subset(full_ds, rights)
            
            left_batch, right_batch = Loader(left_ds, right_ds, args)
            
            if rep == 'NA':
                preds = model(left_batch, right_batch)
            else:
                preds = model(rep(left_batch), rep(right_batch))
                
            _, pred_labels = torch.max(preds, 1)

            predlables = np.append(predlables, pred_labels.cpu().detach().numpy())
            
            batch_labels = y_train[idxs].to(device)
            gt = np.append(gt, batch_labels.cpu().detach().numpy())

            loss = crit(preds, batch_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            losses += loss.item()
            
        if epoch % 20 == 0:
            print('training epoch: ', epoch, ' time: ',time.time()-t0)

    return model
    
    

def predict(model, rep, ds, pair_df, args, record = False):
    model = model.to(args.device)
    if rep != 'NA':
        rep = rep.to(args.device)
    model.eval()
    res = np.empty(0)
    if rep == 'NA':
        optimizer = torch.optim.Adam(list(model.parameters()),lr = args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters())+ list(rep.parameters()),lr = args.lr)
    for idxs in chunks(range(len(pair_df)), args.bs):
        
        optimizer.zero_grad()

        lefts = pair_df.loc[idxs, 'Left'].to_numpy()
        rights = pair_df.loc[idxs, 'Right'].to_numpy()
        
        left_ds = torch.utils.data.Subset(ds,lefts)
        right_ds = torch.utils.data.Subset(ds,rights)
        
        left_batch, right_batch = Loader(left_ds, right_ds, args)
        
        if rep == 'NA':
            preds = model(left_batch, right_batch)
        else:
            preds = model(rep(left_batch), rep(right_batch))
            
        _, pred_labels = torch.max(preds, 1)
        
        res = np.append(res, pred_labels.cpu().detach().numpy())
        
    
    return res


class Args:
    device = 'cuda:0'
    bs = 64
    epochs = 150
    lr = 1e-3
    hid = 64
    save_path = 'results/index_swap/original/'
    max_filters = 15
    method = 'avgdl'
    save_group = 'avgdl'
    splitting = 'pair' ## plan, query (template in tpc-)
    group = 0 ## in tpch and tpcds
    ##
    threshold = 5
    dataset = 'stats'
    
args = Args()
bs = args.bs
import os
save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
device = args.device
hid = args.hid
bs = args.bs
method = args.method

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

pair_df = pd.read_csv('../data/stats/index/pair_df.csv')
long_raw = pd.read_csv('../data/stats/index/long_raw_compiled.csv')

# pair_df = pd.read_csv('../data/tpcds/pair_df_sampled.csv')
# long_raw = pd.DataFrame()
# for i in range(21):
#     file = '../data/tpcds/long_raw_part{}.csv'.format(i)
#     df = pd.read_csv(file)
#     long_raw = long_raw.append(df)
# long_raw.reset_index(drop=True, inplace=True)

# pair_df = pd.read_csv('../data/tpch/pair_ids.csv')
# long_raw = pd.read_csv('../data/tpch/long_raw.csv')

pair_df_swapped = randomSwap(pair_df)
pair_df = pair_df_swapped

from dataset_utils import *

roots, js_nodes, idxs = df2nodes(long_raw)
for i in range(len(roots)):
    roots[i].query_id = i

dat_path = '../data/stats/'
minmax = pd.read_csv(dat_path+ 'column_min_max_vals.csv')
col_min_max = get_col_min_max(minmax)
ds_info = DatasetInfo({})
alias2table = ds_info.alias2table
ds_info.construct_from_plans(roots)
ds_info.get_columns(col_min_max)
costs = get_costs(js_nodes)

train_df,test_df = split_train_test(pair_df, long_raw, args.splitting, args.threshold, args.group, args.dataset)
train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)
y_train = torch.LongTensor(collate_labels(train_df))
y_test = torch.LongTensor(collate_labels(test_df))

full_ds, rep, model = get_dat_model(roots, costs, args)
model = train(model, rep, full_ds, train_df, ds_info, args)

predictions = predict(model, rep, full_ds, train_df, args, record = False)
train_acc, train_f1, train_avg_f1 = compute_score(predictions, y_train.numpy())
print(train_acc, train_f1, train_avg_f1)

predictions = predict(model, rep, full_ds, test_df, args, record = True)
test_acc, test_f1, test_avg_f1 = compute_score(predictions, y_test.numpy())
print(test_acc, test_f1, test_avg_f1)