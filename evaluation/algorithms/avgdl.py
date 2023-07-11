import pandas as pd
import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class AVGDL(nn.Module):
    def __init__(self, embed_dim = 32, hid1_size = 64, hid2_size = 64, max_emb = 1000):
        super(AVGDL, self).__init__()
        self.input_size = embed_dim
        self.hid1_size = hid1_size
        self.hid2_size = hid2_size
        self.embed_dim = embed_dim
        self.lstm1 = nn.LSTM(input_size = embed_dim, hidden_size = hid1_size, \
                             num_layers = 1, batch_first = True)
        self.lstm2 = nn.LSTM(input_size = hid1_size, hidden_size = hid2_size, \
                            num_layers = 1, batch_first = True)
        self.embed = nn.Embedding(max_emb,embed_dim)
        
    def forward(self, batch):
        feature, feat_lens, node_lens = batch.feature, batch.feat_lens, batch.node_lens
        n_trees, n_nodes = feature.size()[:2]
        
        # -1 is no. of elements in a node
        feats = self.embed(feature).view(n_trees*n_nodes,-1, self.embed_dim) 
#         print(feats.shape, feat_lens)
        packed_feat = pack_padded_sequence(feats, feat_lens.view(-1), \
                                           batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm1(packed_feat)
        out,lens = pad_packed_sequence(output, batch_first=True)
        last_items_out = out[torch.arange(n_trees * n_nodes),lens-1].view(n_trees,n_nodes,-1)
        
        packed_nodes = pack_padded_sequence(last_items_out, node_lens, \
                                        batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm2(packed_nodes)
        out,lens = pad_packed_sequence(output, batch_first=True)
        last_items_out = out[torch.arange(n_trees),lens-1].view(n_trees,-1)
        
        return last_items_out


class Batch():
    def __init__(self, feature, feat_lens, node_lens):
        self.feature = feature
        self.feat_lens = feat_lens
        self.node_lens = node_lens
    def to(self, device):
        self.feature = self.feature.to(device)
#         self.feat_lens = self.feat_lens.to(device)
#         self.node_lens = self.node_lens.to(device)
        return self


class Encoding():
    def __init__(self):
        self.str2idx = {}
        self.max_len = 0
        
    def encode_tree(self, root):
        res = []
        def dfs(node):
            nonlocal res
            res.append(self.encode_node(node))
            for child in node.children:
                dfs(child)
        dfs(root)
        return res
        
    def encode_node(self, node):
        res_str = [node.nodeType, node.table, node.index, node.join]
        res_str.extend(self.encode_filters(node.filters))
        res = []
        res = [self.map_idx(ele) for ele in res_str]
        if len(res) > self.max_len:
            self.max_len = len(res)
        return res
    def map_idx(self, string):
        if string is None:
            string = '%None'
        if string not in self.str2idx:
            self.str2idx[string] = len(self.str2idx)
        return self.str2idx[string]
    def encode_filters(self, filters): # list of list of 3
        res = []
        for filt in filters:
            for ele in filt:
                if isinstance(ele, (float, int)):
                    res.append('%NUM')
                else:
                    res.append(ele)
        return res

class AVGDL_Dataset(Dataset):
    def __init__(self, roots, encoding, labels, ds_info):
        self.encoding = encoding
        self.roots = roots
        self.labels = torch.FloatTensor(ds_info.cost_norm.normalize_labels(labels)).reshape(-1,1)
        self.features, self.lenss, self.n_nodes = self.encode(roots)
    
    def encode(self, roots):
        features, lenss, n_nodes = [],[],[]
        for root in roots:
            feature = self.encoding.encode_tree(root)
            features.append(feature)
            lenss.append([len(nf) for nf in feature])
            n_nodes.append(len(feature))
        return features, lenss, n_nodes
            
    def __len__(self):
        return len(self.roots)
    
    def __getitem__(self,idx):
        return self.features[idx], self.lenss[idx], self.n_nodes[idx], self.labels[idx]        

from torch.nn.utils.rnn import pad_sequence
def collate(small_set):
    feats = [s[0] for s in small_set]
    lenss = [s[1] for s in small_set]
    n_nodes = [s[2] for s in small_set]
    labels = [s[3] for s in small_set]
    
    max_lens = max([max(e) for e in lenss])
    max_n_nodes = max(n_nodes)
    fs = []
    for feat in feats:
    	## this 2 lines forces inner lstm seq_len match max amang all in batch
        ftz = [torch.LongTensor(s) for s in feat]
        ftz[0] = nn.ConstantPad1d((0, max_lens - ftz[0].shape[0]), 0)(ftz[0])
        f = pad_sequence(ftz, True, 0)
#         print(f.size())
        fs.append(f)
    
    final_feat = pad_sequence(fs, True, 0)
    lens_tensor = pad_sequence([torch.LongTensor(l) for l in lenss], True, 1)
    return Batch(final_feat, lens_tensor, torch.LongTensor(n_nodes)), torch.cat(labels).view(-1,1)

