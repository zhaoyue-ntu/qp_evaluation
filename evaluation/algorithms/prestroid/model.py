import torch.nn as nn
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation

import numpy as np

JOIN_TYPES = ['Hash Join', 'Nested Loop', 'Merge Join', 'Other Join']
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan", 
              'Bitmap Heap Scan', 'Subquery Scan']
ALL_TYPES = JOIN_TYPES + SCAN_TYPES


class Batch():
    def __init__(self,trees, idxes, masks):
        self.trees = trees
        self.idxes = idxes
        self.masks = masks
    def to(self,dev):
        self.trees = self.trees.to(dev)
        self.idxes = self.idxes.to(dev)
        self.masks = self.masks.to(dev)
        return self


def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    if isinstance(x, tuple):
        return x[0]
    return x

def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.FloatTensor(targets).reshape(-1, 1)
    flat_trees, indexes = prepare_trees(trees, features, left_child, right_child)
    
    length = targets.size()[0]
    max_nodes = flat_trees.size()[-1]
    samples = torch.zeros((length,max_nodes))

    for i, (tree, target) in enumerate(x):
        preorder = preorder_indexes(tree,left_child, right_child)
        sp = subsample(preorder)
        samples[i,sp] = 1

    return Batch(flat_trees, indexes, samples), targets

from TreeConvolution.util import *
def prepare_trees(trees, transformer, left_child, right_child, cuda=False):
    flat_trees = [flatten(x, transformer, left_child, right_child) for x in trees]
    flat_trees = pad_and_combine(flat_trees)
    flat_trees = torch.Tensor(flat_trees)

    # flat trees is now batch x max tree nodes x channels
    flat_trees = flat_trees.transpose(1, 2)

    indexes = [tree_conv_indexes(x, left_child, right_child) for x in trees]
    indexes = pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()

    return (flat_trees, indexes)



class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg

def is_join(node):
    return len(node.children) >= 2

def is_scan(node):
    if node.children == []:
        return True
    return node.nodeType in SCAN_TYPES


class TreeBuilder:
    def __init__(self, stats_extractor, relations):
        self.stats = stats_extractor
        self.relations = sorted(relations, key=lambda x: len(x), reverse=True)

    def featurize_join(self, node):
        assert is_join(node)
        arr = np.zeros(len(ALL_TYPES) + len(self.relations) + 1)
        if node.nodeType in ALL_TYPES:
            arr[ALL_TYPES.index(node.nodeType)] = 1
        else:
            arr[ALL_TYPES.index("Other Join")] = 1
            
        if node.table is not None and node.table in self.relations:
            arr[len(ALL_TYPES)+self.relations.index(node.table)] = 1
        else: # other table or no table
            arr[-1] = 1       
        return np.concatenate((arr, self.stats(node)))

    def featurize_scan(self, node):
        assert is_scan(node)
        arr = np.zeros(len(ALL_TYPES) + len(self.relations) + 1)
        if node.nodeType in ALL_TYPES:
            arr[ALL_TYPES.index(node.nodeType)] = 1
        if node.table is not None and node.table in self.relations:
            arr[len(ALL_TYPES)+self.relations.index(node.table)] = 1
        else:
            arr[-1] = 1 
        
        return (np.concatenate((arr, self.stats(node))),
                node.table)

    def plan_to_feature_tree(self, root):
        children = root.children
       
        if is_join(root):
            assert len(children) >= 2
            my_vec = self.featurize_join(root)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            
            if len(children) == 3:
                mid = self.plan_to_feature_tree(children[2])
    #             print(mid)
                return (my_vec, mid, (my_vec, left, right))
    
            return (my_vec, left, right)

        if is_scan(root):
            if root.children == []:
                return self.featurize_scan(root)
            else:
    #           select only the first node, which is the root of the subquery
                if isinstance(self.plan_to_feature_tree(root.children[0]), tuple):
    #               when subquery is a complex query
                    my_vec = self.plan_to_feature_tree(root.children[0])[0]
                else:
    #               when subquery has only one scan
                    my_vec = self.plan_to_feature_tree(root.children[0])
                my_vec[-1] = 1
                return my_vec

        
        if len(children) == 1:
            return self.plan_to_feature_tree(children[0])


        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))



def norm(x, lo, hi):
    return (np.log(x + 1) - lo) / (hi - lo)

def get_buffer_count_for_leaf(leaf, buffers):
    total = 0
    if leaf.table:
        total += buffers.get(leaf.table, 0)

    if leaf.index:
        total += buffers.get(leaf.index, 0)

    return total

class StatExtractor:
    def __init__(self, fields, mins, maxs):
        self.__fields = fields
        self.__mins = mins
        self.__maxs = maxs

    def __call__(self, inp):
        res = []
        for f, lo, hi in zip([inp.buffers, inp.cost_est, inp.card_est], self.__mins, self.__maxs):
            if not f:
                res.append(0)
            else:
                res.append(norm(f, lo, hi))
        return res


def get_plan_stats(roots):
    costs = []
    rows = []
   
    def recurse(n):
        costs.append(n.cost_est)
        rows.append(n.card_est)

        if n.children != []:
            for child in n.children:
                recurse(child)

    for root in roots:
        recurse(root)

    costs = np.array(costs)
    rows = np.array(rows)
    
    costs = np.log(costs + 1)
    rows = np.log(rows + 1)

    costs_min = np.min(costs)
    costs_max = np.max(costs)
    rows_min = np.min(rows)
    rows_max = np.max(rows)

#     if len(bufs) != 0:
    return StatExtractor(
        ["Plan Rows"],
        [rows_min],
        [rows_max]
    )
        

def get_all_relations(data):
    all_rels = []
    
    def recurse(root):
        if root.table:
            yield root.table

        if root.children != []:
            for child in root.children:
                yield from recurse(child)

    for root in data:
        all_rels.extend(list(recurse(root)))
        
    return set(all_rels)

def get_featurized_trees(data):
    all_rels = get_all_relations(data)
    stats_extractor = get_plan_stats(data)

    t = TreeBuilder(stats_extractor, all_rels)
    trees = []

    for root in data:
        tree = t.plan_to_feature_tree(root)
        trees.append(tree)
            
    return trees


class TreeFeaturizer:
    def __init__(self):
        self.__tree_builder = None

    def fit(self, trees):

        all_rels = get_all_relations(trees)
        stats_extractor = get_plan_stats(trees)
        self.__tree_builder = TreeBuilder(stats_extractor, all_rels)

    def transform(self, trees):

        return [self.__tree_builder.plan_to_feature_tree(x) for x in trees]

    def num_operators(self):
        return len(ALL_TYPES)






def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    if isinstance(x, tuple):
        return x[0]
    return x

class Prestroid(nn.Module):
    def __init__(self, in_channels):
        super(Prestroid, self).__init__()
        self.in_channels = in_channels

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            ## to modify
#             DynamicPooling(),
        )

    def in_channels(self):
        return self.in_channels
        
    def forward(self, x):
        feats = self.tree_conv((x.trees, x.idxes))[0]
        
        masks = x.masks
        n_per_mask = torch.sum(masks,dim=1)
        mp = masks.unsqueeze(1).repeat(1,64,1)
        masked_feats = feats*mp
        
        return torch.max(masked_feats, dim=2).values


from collections import deque
def subsample(preorder_indexes, K = 5, N = 15):
    idxs = preorder_indexes
    if isinstance(idxs, int):
        return [1]
    res = []
    subtree_size = 0
    
    toVisit = deque()
    toVisit.append(idxs)
    
    def add_node(triplet):
        nonlocal res
        nonlocal subtree_size
        if isinstance(triplet, int):
            res.append(triplet)
            subtree_size += 1
        else:
            res.append(triplet[0])
            subtree_size += 1
    while toVisit:
        triplet = toVisit.popleft()
        if not isinstance(triplet, int):
            toVisit.append(triplet[1])
            toVisit.append(triplet[2])
        if len(res) >= K: 
            break
        if subtree_size > N: 
            subtree_size = 0
            continue
        add_node(triplet)
        subtree_size += 1
    return res
        