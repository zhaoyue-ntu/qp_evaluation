import pandas as pd
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import json
from datetime import datetime
import time
import dgl
from collections import deque
import copy
# what we need:
# column id
# finding other columns from the same table
# table:
#    n x p x 5 (col_id, F(c))
#    n: number of cols in a table
#    p: number of predicates in a col

def get_rtos_dataset(roots, costs, ds_info, encoding):
    labels = ds_info.cost_norm.normalize_labels(costs)
    dataset = [plan2feature(node,encoding,ds_info) for node in roots]
    return list(zip(dataset, labels))

class Encoding:
    def __init__(self, ds_info):
        self.column_min_max_vals = ds_info.column_min_max_vals
        
        self.op2idx = {'>':0, '=':1, '<':2}
        
        self.table2alias = ds_info.table2alias
        self.alias2table = ds_info.alias2table
        
        col2idx = {'NA':0}
        idx = 1
        col2table = {}
        alias2cols = {}
        for col in ds_info.columns:
            col2idx[col] = idx
            idx += 1
            alias, _ = col.split('.')
            col2table[col] = alias
            if alias not in alias2cols:
                alias2cols[alias] = [col]
            else:
                alias2cols[alias].append(col)
        self.col2idx = col2idx
        self.col2table = col2table
        self.alias2cols = alias2cols
        
        self.max_table_col = max([len(t) for t in alias2cols])
        
    def is_number(self,s):
        try:
            float(s)
            return True
        except:
            return False
    
    def normalize_val(self, column, val):
        if column not in self.column_min_max_vals or (not self.is_number(val)):
            # print('column {} not in col_min_max'.format(column))
            return 0.
        mini, maxi = self.column_min_max_vals[column]
        val_norm = 0.
        val = float(val)
        if maxi > mini:
            val_norm = (val-mini) / (maxi-mini)
        return val_norm
    
    def encode_col(self, col):
        if col not in self.col2idx:
            self.col2idx[col] = len(self.col2idx)
        return self.col2idx[col]


def node2col(node, encoding): # for intermediate nodes
    l, r = node.join.split(' = ')
    lid = encoding.encode_col(l)
    rid = encoding.encode_col(r)
    return np.array([lid,1,0,0,0]), \
            np.array([rid,1,0,0,0]) # id, F(C)


# table:
#    n x p x 5 (col_id, F(c))
#    n: number of cols in a table
#    p: number of predicates in a col
# n is unnecessary since we do max pooling regardless
# multiple predicate in same column is treated the same as 
# different columns in a table
# so just p x 5
def node2table(node, ds_info, encoding, null=False): # for scan nodes
    n = encoding.max_table_col
    p = ds_info.max_filters
    res = np.zeros((p, 5))
    if null:
        return res
    for i,filt in enumerate(node.filters):
        col = encoding.encode_col(filt[0])
        val = encoding.normalize_val(filt[0],filt[2])
#         print(filt,val)
        if filt[1] == '=':
            line = np.array([col, 0., val+1., 0., 0.])
        elif filt[1] == '>':
            line = np.array([col, 0., 0., val, 0.])
        else:
            line = np.array([col, 0., 0., 0., 1-val])
        res[i,:] = line
    num = len(node.filters)
    return res


from collections import deque
def plan2feature(root, encoding, ds_info):
    ## graph rep
#     root = copy.deepcopy(root)
    srcs = []
    trgs = []
    parent_id = {}
    node_id = {}
    ## features
    table_feat = []
    col_left = []
    col_right = []

    num_filters = []
    # bfs
    toVisit = deque()
    toVisit.append((root,0))
    idx = 0
    node_id[root] = idx
    while toVisit:
        node,pid = toVisit.pop()

        if len(node.children) == 1: # transparent node, pass down
            toVisit.append((node.children[0],pid))
            if idx != 0:
                parent_id[node.children[0]] = pid
            continue
        node_id[node] = idx
        idx += 1   
        if node in parent_id:
            srcs.append(node_id[node])
            trgs.append(parent_id[node])

        if len(node.children) == 0: # scan
            table_feat.append(node2table(node, ds_info, encoding))
            num_filters.append(max(len(node.filters),1))
            
            col_left.append(np.zeros(5))
            col_right.append(np.zeros(5))
            
        else: # join
#             assert(node.join != 'NA')
            table_feat.append(node2table(node, ds_info, encoding, null=True))
            num_filters.append(1)
            
            if node.join == 'NA':
                col_left.append(np.ones(5))
                col_right.append(np.ones(5))
            else:
                l, r = node2col(node, encoding)
                col_left.append(l)
                col_right.append(r)
            for i,child in enumerate(node.children):
                child = node.children[i]
                if i >= 2: # change to left deep
                    # print(toVisit[-1][0].children)
                    del node.children[i]
                    toVisit[-1][0].children.append(child)
                else:
                    toVisit.append((child,node_id[node]))
                    parent_id[child] = node_id[node]
                
    
    srcs = torch.tensor(srcs,dtype=torch.int64)
    trgs = torch.tensor(trgs,dtype=torch.int64)
    table_feat = torch.Tensor(np.array(table_feat))
    col_left = torch.Tensor(np.array(col_left))
    col_right = torch.Tensor(np.array(col_right))
    idx = torch.Tensor([idx])
    g = dgl.graph((srcs,trgs),num_nodes=idx)
    g.ndata['table_feat'] = table_feat
    g.ndata['col_left'] = col_left
    g.ndata['col_right'] = col_right
    g.ndata['num_filters'] = torch.tensor(np.array(num_filters),dtype=torch.int64)
    return g



class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        
        self.h_size = h_size
        
        # W for columns and U for tables
        self.W_iou_left = nn.Linear(x_size, 3 * h_size, bias=False)
        self.W_iou_right = nn.Linear(x_size, 3 * h_size, bias=False)
        self.W_f_left = nn.Linear(x_size, h_size, bias=False)
        self.W_f_right = nn.Linear(x_size, h_size, bias=False)
        
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
#         print('h before cat ',nodes.mailbox['h'].size())
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
#         print('hcat ',h_cat.size())
        
#         print(f.size(), c.size())
        # equation (2)
#         print('ux, ',self.U_f(h_cat).view(*nodes.mailbox['h'].size()).size())
        b, n, d = nodes.mailbox['h'].size()
        eq2 = nodes.data['wfx'].view(-1,self.h_size) + \
              torch.sum(self.U_f(h_cat).view(b,n,d),dim = 1)
        f = th.sigmoid(eq2).view(b,1,d)
#         print('f: ',f.size())
#         print('c: ',nodes.mailbox['c'].size())
        # second term of equation (5)
        c = th.sum(f * nodes.mailbox['c'], 1)

        
        return {'wioux': nodes.data['wioux'] + \
                self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['wioux'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # equation (5)
#         print('iou',i.size(),o.size())
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * th.tanh(c)
#         print(c)
        return {'h' : h, 'c' : c}

import dgl.function as fn
class TreeLSTM(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_cols,
                 h_size,
                 pooling = 'max'):
        super(TreeLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hs = emb_dim // 4
        self.embedding = nn.Embedding(num_cols, emb_dim) # emb_dim is 4 x hs

        self.cell = TreeLSTMCell(emb_dim // 4, h_size)
        self.h_size = h_size
        
        self.pooling = pooling
    def forward(self, batch):

        g = batch.graph
        # to heterogenous graph
        device = batch.col_left.device
        g = dgl.graph(g.edges(),num_nodes=batch.graph.num_nodes()).to(device)
        # feed embedding (columns) (intermediate nodes impt)
        ## get W x h_betas TODO
        bs = batch.graph.num_nodes()
        
        col_id, fc = torch.split(batch.col_left, (1,4), dim=-1)
        cols = self.embedding(col_id.long()).view(bs, self.hs, 4) # bs x 256
        mc_left = torch.bmm(cols, fc.unsqueeze(-1)).squeeze(-1) # bs x 64
        
        col_id, fc = torch.split(batch.col_right, (1,4), dim=-1)
        cols = self.embedding(col_id.long()).view(bs, self.hs, 4) # bs x 256
        mc_right = torch.bmm(cols, fc.unsqueeze(-1)).squeeze(-1) # bs x 64

        wioux_left = self.cell.W_iou_left(mc_left)
        wioux_right = self.cell.W_iou_right(mc_right)
        wfx_left = self.cell.W_f_left(mc_left)
        wfx_right = self.cell.W_f_right(mc_right)
        g.ndata['wioux'] = wioux_left + wioux_right # separate from updatable h, since nvr updating
        g.ndata['wfx'] = wfx_left + wfx_right
#         print(embeds.size())

        ## get table things TODO, can actually put in h
#         iou_left = self.cell.W_iou_left(self.dropout(embeds_left)) \
#                                 * batch.mask.float().unsqueeze(-1)
        # b x n x p
        col_id, fc = torch.split(batch.table_feat, (1,4), dim=-1)
        cols = self.embedding(col_id.long()).view(-1, self.hs, 4) # -1 is bs x n
        mcs = torch.bmm(cols, fc.view(-1,4,1)).view(bs, -1, self.hs) # bs x n x hs
        if self.pooling == 'max':
            mc = torch.max(mcs, dim=1)
        else:
            mc = torch.sum(mcs, dim=1) / batch.num_filters.view(-1,1)
#         print('mc: ',mc.size())
        g.ndata['h'] = mc
        # table things (preds) must put inside reduce_fc, 
        # since intermediate nodes only getting them later,
        # column (joins) can put outside though
        
        g.ndata['c'] = th.zeros((bs, self.h_size)).to(device)
        # propagate
        traversal_order = dgl.topological_nodes_generator(g)
        trv_order = [trv.to(device) for trv in traversal_order]
        g.prop_nodes(trv_order[1:], # skip over leaves
                     message_func=self.cell.message_func,
                     reduce_func=self.cell.reduce_func,
                     apply_node_func=self.cell.apply_node_func)
#         dgl.prop_nodes_topo(g,
#                             message_func=self.cell.message_func,
#                             reduce_func=self.cell.reduce_func,
#                             apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = g.ndata.pop('h')
#         h = self.dropout(h)
        return h[batch.root_ids]        


class Batch():
    def __init__(self, batch_trees, root_ids):
        self.graph = batch_trees
        self.col_left = batch_trees.ndata['col_left']
        self.col_right = batch_trees.ndata['col_right']
        self.table_feat = batch_trees.ndata['table_feat']
        self.root_ids = root_ids
        self.num_filters = batch_trees.ndata['num_filters']
    
    def to(self, device):
        self.graph = self.graph.to(device)
        self.col_left = self.col_left.to(device)
        self.col_right = self.col_right.to(device)
        self.table_feat, self.root_ids = self.table_feat.to(device), self.root_ids.to(device)
        self.num_filters = self.num_filters.to(device)
        return self     
    
def batcher(batch):
    y = [b[1] for b in batch]
    batch = [b[0] for b in batch]
    batch_trees = dgl.batch(batch)
    n_nodes = [g.num_nodes() for g in batch]
    root_ids = np.cumsum([0]+n_nodes[:-1])
    
    return Batch(batch_trees, torch.tensor(root_ids,dtype=torch.int64)), torch.FloatTensor(y).view(-1,1)
