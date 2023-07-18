import pandas as pd
import numpy as np
import torch
import json

import sys
sys.path.append('../evaluation/')


df_list = []
for arm in range(49):
    df_list.append(pd.read_csv('../data/imdb/bao/plans/job_ext_arm{}.csv'.format(arm)))

import pickle
with open('../data/imdb/bao/plans/bao_dat.pkl','rb') as inp:
    dat = pickle.load(inp)
planss = dat['planss']
latencies = dat['latencies']
rootss = dat['rootss']
del dat

## if pickle not usable
# planss = []
# latencies = []
# rootss = []
# for i in range(49):
#     plans = [json.loads(plan) for plan in df_list[i]['json']]
#     roots = [traversePlan(pl) for pl in plans]
    
#     rootss.append(roots)
#     planss.append(plans)
#     latency = [json.loads(plan)['Execution Time'] for plan in df_list[i]['json']]
#     latencies.append(latency)

from dataset_utils import *
all_roots = sum(rootss,[])
ds_info = DatasetInfo({})
ds_info.construct_from_plans(all_roots)

minmax = pd.read_csv('../data/imdb/column_min_max_vals.csv')
col_min_max = get_col_min_max(minmax)
ds_info.get_columns(col_min_max)

import random

## Main Module
## It's an offline simulation to avoid executing the same query plans millions of times
class BanditOptimizer():
    def __init__(self, planss, rootss, latencies, look_back = 800, N = 100, freq = 100):
        ## system settings
        self.N = N
        self.look_back = look_back
        self.freq = freq
        ##
        
        self.planss = planss
        self.rootss = rootss
        self.latencies = latencies
        
        self.arms = len(self.latencies)
        self.total = len(self.latencies[0])
        self.cur_query = freq
        self.selections = [0 for i in range(freq)]
        self.tm = [0 for i in range(freq)] # inference
        self.tl = [0 for i in range(freq)] # pre-process
        self.tr = [0 for i in range(freq)] # train
        self.exe_time = []
        ## record results
        # training time
        # inference time
        # query execution time
        # creating ds time
        
        random.seed(42)
        self.sample_ids = []
        for i in range(0,self.total//self.freq+1):
            left = max(0, i*self.freq-self.look_back)
            right = (i+1)*self.freq
            ids = random.choices(range(left, right), k=self.N)
            self.sample_ids.append(ids)
        self.spl = 0
    
    def get_execution_time(self):
        exe_time = []
        for i,sel in enumerate(self.selections):
            exe_time[i] = self.latencies[i][sel]
        self.exe_time = exe_time
        return exe_time
        
    def initial_data(self):
        return self.rootss[0][:self.freq], self.latencies[0][:self.freq]
    
    ## thompson sampling
    ## sample with replacement from exp to train model
    def sample_data(self):
        if self.cur_query == self.freq:
            return self.initial_data()
        
#         left = max(0, self.cur_query-self.look_back)
#         sample_ids = random.choices(range(left,self.cur_query),k=self.N)
        if self.spl >= len(self.sample_ids):
            print('Shld alr be done, pls check')
            return None
        sample_ids = self.sample_ids[self.spl]
        self.spl +=1
        
        roots = []
        lats = []
        for idx in sample_ids:
            sel = self.selections[idx]
            roots.append(self.rootss[sel][idx])
            lats.append(self.latencies[sel][idx])
        return roots, lats
    
#     choose next N query plans and add to experience
    ## choose next (freq) query plans and add to experience
    
    def select_plans(self, model, get_batch):
        sels = []
        right = min(self.total,self.cur_query+self.freq)
        qids = range(self.cur_query, right)
        tm = []
        tl = []
        for qid in qids:
            roots = [self.rootss[i][qid] for i in range(self.arms)]
            lats = [self.latencies[i][qid] for i in range(self.arms)]
            
            t0 = time.time()
            batch = get_batch(roots, lats)
            t1 = time.time()
            out = model(batch).squeeze()
            t2 = time.time()
            tm.append(t2-t1)
            tl.append(t1-t0)
            
            sels.append( out.detach().cpu().argmin().numpy().item() )
            
            del batch
            
        self.selections += sels
        self.cur_query = right
        self.tm += tm
        self.tl += tl
        print('Model Time: {}, Preprocessing Time: {}'.format(sum(tm), sum(tl)))

        ## to get some reference numbers
        latss = [[self.latencies[i][qid] for i in range(self.arms)] for qid in qids]
        best_lats = 0
        post_lats = 0
        sel_lats = 0
        for i,qid in enumerate(qids):
            lats = [self.latencies[k][qid] for k in range(self.arms)]
            post_lats += self.latencies[0][qid] / 1000
            best_lats += min(lats) / 1000
            sel_lats += self.latencies[sels[i]][qid] / 1000
        print('Best Time: {}, Post Time: {}, Sel Time: {}'.format(best_lats, \
                                                post_lats, sel_lats))
        
        return self.selections
    
    def train_time(self, tr):
        print(len(self.tr))
        self.tr.append(tr)
        remain_len = min(self.freq-1, self.total-len(self.tr)) 
        self.tr += [0 for a in range(remain_len)]
        
def get_custom(latencies, df):
    total_lats = []
    execution_lats = []
    for i, row in df.iterrows():
        sel = row['Selections']
        lat = latencies[sel][i] / 1000
        execution_lats.append(lat)
        total = lat + row['Train Time'] + \
            row['Inf Time'] + row['Preprocess Time']
        total_lats.append(total)
    return total_lats, execution_lats


from algorithms.avgdl import AVGDL_Dataset, AVGDL
from algorithms.avgdl import Encoding as avgdl_Encoding
from algorithms.avgdl import DataLoader as avgdl_loader
from algorithms.avgdl import collate as avgdl_collate

class Args:
    device = 'cuda:0'
    bs = 128
    epochs = 200
    lr = 1e-3
    hid = 64
    save_path = 'results/bao/avgdl/'
args = Args()
import os
save_path = args.save_path 
if not os.path.exists(save_path):
    os.makedirs(save_path)


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

from trainer import Prediction,train
from torch import nn
encoding = avgdl_Encoding()
avgdl = AVGDL(32, 64, 64)
prediction = Prediction(64, args.hid)
model = nn.Sequential(avgdl, prediction)
_ = model.to(args.device)
def construct_loader(ds_info):
    def get_loader(roots, costs):
        _train_roots = roots
        ds = AVGDL_Dataset(_train_roots, encoding, costs, ds_info)
        return avgdl_loader(ds, batch_size = len(roots), collate_fn=avgdl_collate, shuffle=False)
    return get_loader
def construct_batch(get_loader, args):
    def get_batch(roots, costs):
        loader = get_loader(roots,costs)
        return next(iter(loader))[0].to(args.device)
    return get_batch
get_loader = construct_loader(ds_info)
get_batch = construct_batch(get_loader,args)

N = 400
look_back = 800
freq = 100
bo_agent = BanditOptimizer(planss, rootss, latencies,look_back=look_back,N=N,freq=freq)


for steps in range(len(latencies[0])//freq):
    t0 = time.time()
    dat = bo_agent.sample_data()
    loader = get_loader(*dat)
    train(model, loader, loader, dat[1], ds_info, args, prints=False,record=False)
    bo_agent.train_time(time.time()-t0)
    print('Training Time: {}'.format(time.time()-t0))
    bo_agent.select_plans(model,get_batch)


res = df_list[0].copy()
del res['json']
res['Train Time'] = bo_agent.tr
res['Inf Time'] = bo_agent.tm
res['Preprocess Time'] = bo_agent.tl
res['Selections'] = bo_agent.selections

arms = len(latencies)
length = len(latencies[0])
best_sels = []
best_lats = []

worst_lats = []

for i in range(length):
    lats = [latencies[k][i] for k in range(arms)]
    mini = min(lats)
    best_lats.append(mini)
    best_sels.append(lats.index(mini))
    worst_lats.append(max(lats))
    
queries_complete = range(length)

best = np.cumsum(best_lats) / 1000 / 60
post = np.cumsum(latencies[0]) / 1000 / 60

total_final = np.cumsum(total_time) / 60
exe_final = np.cumsum(exe)  / 60

print('Best Possible | Postgres | Total | Query Time')
print(best[-1], post[-1], total_final[-1], exe_final[-1])

# from matplotlib import pyplot as plt
# total_time, exe = get_custom(latencies, res)

# total_final = np.cumsum(total_time) / 60
# exe_final = np.cumsum(exe)  / 60


# fig, ax = plt.subplots(1, 1, constrained_layout=True,figsize=[6,6])

# ax.plot(post, queries_complete, label="PostgreSQL", lw=3)

# ax.plot(total_final, queries_complete, label='Exe final', lw=3)

# ax.plot(exe_final, queries_complete, label='Exe queries',lw=3)

# # ax.plot(simple_cum_runtime, queries_complete,'--', label='QF (simple)', lw=3, color='#9467bd',alpha=0.6)

# # ax.plot(hist_cum_runtime, queries_complete,'--', label = 'QF (no-hist)', lw=3, color='#d62728')

# ax.plot(best, queries_complete,':', label='Optimal', lw=3, color='tab:gray')


# # ax.plot(my_off_new_cum, queries_complete)

# ax.legend(loc='upper left', prop={'size': 15})


# ax.set_xlabel("Time (m)",fontsize=23)
# ax.set_ylabel("Queries complete",fontsize=25)
# # ax.set_title("Number of Queries Complete over Time")

# ax.grid(linestyle="--", linewidth=1)
# ax.legend(loc='upper left', prop={'size': 15})
# # fig.savefig("queries_vs_time_with_ablation_v4.pdf")