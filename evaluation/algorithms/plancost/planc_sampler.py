from random import shuffle
from copy import deepcopy
import torch

def grouping(data):
    """
        Groups the queries by their query plan structure

        Args:
        - data: a list of root nodes, each being a query from the dataset

        Returns:
        - enum    : a list of same length as data, containing the group indexes for each query in data
        - counter : number of distinct groups/templates
    """
    def hash(root):
        res = root.nodeType
        if root.children != []:
            for chld in root.children:
                res = res + ', ' + hash(chld)
        return res
    counter = 0
    string_hash = []
    enum = []
    for root in data:
        string = hash(root)
        #print(string)
        try:
            idx = string_hash.index(string)
            enum.append(idx)
        except:
            idx = counter
            counter += 1
            enum.append(idx)
            string_hash.append(string)
    assert(counter>0)
    return enum, counter


class Batch():
    def __init__(self, ins):
        self.ins = ins
    def to(self, dev):
        # for i in range(len(self.ins)):
        #     self.ins[i]["feat_vec"] = self.ins[i]["feat_vec"].to(dev)
        return self

def collate_fn(ls):
    ins = []
    costs = []
    for x, y in ls:
        ins.append(x)
        costs.append(y)
        
    costs = torch.FloatTensor(costs)
    return Batch(ins), costs

        

class BatchSampler():
    def __init__(self, sampling_ind_list, batch_size, drop_last = False):
        self.sampler_order = sampling_ind_list
        self.batch_size = batch_size
        self.drop_last = drop_last
        
    def __iter__(self):
        batch = []
        for idxs in self.sampler_order:
            inds = deepcopy(idxs)
            shuffle(inds)
            while inds != []:
                batch = inds[:min(self.batch_size, len(inds))]
                del inds[:min(self.batch_size, len(inds))]
                if not self.drop_last or len(batch) == self.batch_size:
                    yield batch
                batch = []
            
    def __len__(self):
        if self.drop_last:
            return sum([(len(inds) // self.batch_size) for inds in self.sampler_order])
        else:
            return sum([(len(inds) // self.batch_size + 1) for inds in self.sampler_order])
