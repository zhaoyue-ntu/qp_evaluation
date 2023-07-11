import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class Encoding:
    def __init__(self, ds_info):
        self.column_min_max_vals = ds_info.column_min_max_vals
        self.col2idx = {'NA':0}
        self.op2idx = {'>':0, '=':1, '<':2}
        self.type2idx = {'NA':0}
        self.join2idx = {'NA':0}

        self.table2idx = {'NA':0}
    
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        
    def normalize_val(self, column, val):
        if column not in self.column_min_max_vals or (not is_number(val)):
            # print('column {} not in col_min_max'.format(column))
            return 0.
        mini, maxi = self.column_min_max_vals[column]
        val_norm = 0.
        val = float(val)
        if maxi > mini:
            val_norm = (val-mini) / (maxi-mini)
        return val_norm
    
    def encode_join(self, join):
        if join not in self.join2idx:
            self.join2idx[join] = len(self.join2idx)
        return self.join2idx[join]
    
    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
        return self.table2idx[table]

    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
        return self.type2idx[nodeType]
    
    def encode_op(self, op):
        if op not in self.op2idx:
            self.op2idx[op] = len(self.op2idx)
        return self.op2idx[op]
    
    def encode_col(self, col):
        if col not in self.col2idx:
            self.col2idx[col] = len(self.col2idx)
        return self.col2idx[col]

# def node2feature(node, encoding, max_filters = 10):
#     num_filter = len(node.filters)
#     n = max(num_filter, 1)
#     pad = np.zeros((3, max_filters-n)) # leave at least 1 for convenience
#     if num_filter > 0:
#         filts = np.array(
#             [[encoding.encode_col(col), 
#               encoding.encode_op(op), 
#               encoding.normalize_val(col, val)] for col,op,val in node.filters]
#         ).T
#     else:
#         filts = np.array([[0.,0.,0.]]).T
#     ## max_filters x3, get back with reshape(3, max_filters)
#     filts = np.concatenate((filts, pad), axis=1)
#     filts = filts.flatten() 
#     mask = np.zeros(max_filters)
#     mask[:n] = 1
#     type_join = np.array([encoding.encode_type(node.nodeType), encoding.encode_join(node.join)])
#     table = np.array([encoding.encode_table(node.table)])
#     return np.concatenate((type_join, filts, mask, table))        

def node2feature(node, encoding, max_filters = 10, hist_file = None, table_sample = None):
    num_filter = len(node.filters)
    n = max(num_filter, 1)
    pad = np.zeros((3, max_filters-n)) # leave at least 1 for convenience
    if num_filter > 0:
        filts = np.array(
            [[encoding.encode_col(col), 
              encoding.encode_op(op), 
              encoding.normalize_val(col, val)] for col,op,val in node.filters]
        ).T
    else:
        filts = np.array([[0.,0.,0.]]).T
    ## max_filters x3, get back with reshape(3, max_filters)
    filts = np.concatenate((filts, pad), axis=1)
    filts = filts.flatten() 
    mask = np.zeros(max_filters)
    mask[:n] = 1
    type_join = np.array([encoding.encode_type(node.nodeType), encoding.encode_join(node.join)])
    table = np.array([encoding.encode_table(node.table)])
    
    ## add db stats
    
    cur_rep = np.concatenate((type_join, filts, mask, table))   
    if hist_file is not None:
        hists = filters2Hist(hist_file, node.filters, encoding, max_filters)
        cur_rep = np.concatenate((cur_rep,hists))
    if table_sample is not None:
        if node.table in table_sample[node.query_id]:
            sample = table_sample[node.query_id][node.table]
        else:
            sample = np.zeros(1000)
        cur_rep = np.concatenate((cur_rep,sample))
    return cur_rep

def filters2Hist(hist_file, filters, encoding, max_filters):
    buckets = len(hist_file['bins'][0]) 
    empty = np.zeros(buckets - 1)
    ress = np.zeros((max_filters, buckets-1))
    table_cols = set(hist_file['table_column'])
    for i,(col,op,val) in enumerate(filters):
        if not is_number(val): continue
        val = float(val)
        if (col == 'NA') or (col not in table_cols):
            ress[i] = empty
            continue
        bins = hist_file.loc[hist_file['table_column']==col,'bins'].item()
        
        left = 0
        right = len(bins)-1
        for j in range(len(bins)):
            if bins[j]<val:
                left = j
            if bins[j]>val:
                right = j
                break

        res = np.zeros(len(bins)-1)

        if op == '=':
            res[left:right] = 1
        elif op == '<':
            res[:left] = 1
        elif op == '>':
            res[right:] = 1
        ress[i] = res
    
    ress = ress.flatten()
    return ress   

def get_job_table_sample(workload_file_name):

    with open(workload_file_name + "_samples.npy",'rb') as f:
        table_sample = np.load(f,allow_pickle=True)
        
    return table_sample


def get_hist_file(hist_path, bin_number = 50):
    hist_file = pd.read_csv(hist_path)
    # for i in range(len(hist_file)):
    #     freq = hist_file['freq'][i]
    #     freq_np = np.frombuffer(bytes.fromhex(freq), dtype=np.float)
    #     hist_file['freq'][i] = freq_np

    # table_column = []
    # for i in range(len(hist_file)):
    #     table = hist_file['table'][i]
    #     col = hist_file['column'][i]
    #     table_alias = ''.join([tok[0] for tok in table.split('_')])
    #     if table == 'movie_info_idx': table_alias = 'mi_idx'
    #     combine = '.'.join([table_alias,col])
    #     table_column.append(combine)
    # hist_file['table_column'] = table_column

    for rid in range(len(hist_file)):
        hist_file['bins'][rid] = \
            [float(i) for i in hist_file['bins'][rid][1:-1].split(' ') if len(i)>0]

    if bin_number != 50:
        raise "only 50 stored to save storage"
        # hist_file = re_bin(hist_file, bin_number)

    return hist_file


def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        bins = freq2bin(freq,target_number)
        hist_file['bins'][i] = bins
    return hist_file

def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq)-1
    
    step = 1. / target_number
    mini = 0
    while freq[mini+1]==0:
        mini+=1
    pointer = mini+1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi+1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1
    
    if len(res_pos)==target_number: res_pos.append(maxi)
    
    return res_pos



class QueryFormerDataset(Dataset):
    def __init__(self, nodes, labels, encoding, ds_info, max_filters=10, \
            max_node=20, rel_pos_max=10, hist_file=None, table_sample=None, query_ids=None):

        self.encoding = encoding

        self.hist_file = hist_file
        self.table_sample = table_sample

        self.max_filters = max_filters
        self.max_node = max_node 
        self.rel_pos_max = rel_pos_max
        
        self.length = len(nodes)
        if query_ids is None:
            for i in range(len(nodes)):
                nodes[i].query_id = i        
        else:
            for i in range(len(nodes)):
                nodes[i].query_id = query_ids[i]

        self.nodes = nodes
        self.ds_info = ds_info
        
        self.costs = labels
        self.cost_labels = torch.FloatTensor(ds_info.cost_norm.normalize_labels(labels)).reshape(-1,1)


        self.collated_dicts = [self.pre_collate(self.node2dict(node)) for node in nodes]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        return self.collated_dicts[idx], self.cost_labels[idx]
      
    ## pre-process first half of old collator
    def pre_collate(self, the_dict):
        ## input is the 'dict'
        x = pad_2d_unsqueeze(the_dict['features'], self.max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= self.rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, self.max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, self.max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], self.max_node)
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, node):

        adj_list, num_child, features = self.topo_sort(node)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(features),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),          
        }
    
    def topo_sort(self, root_node):
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            features.append(node2feature(node, self.encoding, self.max_filters, \
                self.hist_file, self.table_sample))
            num_child.append(len(node.children))
            for child in node.children:
                child.query_id = node.query_id

                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1

        return adj_list, num_child, features

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 


def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j: 
                M[i][j] = 0
            elif M[i][j] == 0: 
                # M[i][j] = 510
                M[i][j] = 60
    
    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k]+M[k][j])
    return M


def pad_1d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # dont know why add 1, comment out first
#    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, attn_bias, rel_pos, heights, x, y):
        super(Batch, self).__init__()
        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos
        
    def to(self, device):
        self.heights = self.heights.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)

        return self

    def __len__(self):
        return self.y.size(0)


def collator(small_set):
    y = [s[1] for s in small_set]
    xs = [s[0]['x'] for s in small_set]
    
    num_graph = len(y)
    x = torch.cat(xs)
    y = torch.cat(y).view(-1,1)
    attn_bias = torch.cat([s[0]['attn_bias'] for s in small_set])
    rel_pos = torch.cat([s[0]['rel_pos'] for s in small_set])
    heights = torch.cat([s[0]['heights'] for s in small_set])
    
    return Batch(attn_bias, rel_pos, heights, x, y), y






















