import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import json
# from pyparsing import nestedExpr
from datetime import datetime
import time
import copy

class Encoding:
    def __init__(self, ds_info):
        self.column_min_max_vals = ds_info.column_min_max_vals
        self.col2idx = {'NA':0}
        self.op2idx = {'>':0, '=':1, '<':2}
        self.type2idx = {'NA':0}
        self.join2idx = {'NA':0}
        
        self.index2idx = {'NA':0}

        self.table2idx = {'NA':0}
    
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
    
    def encode_index(self, index):
        if index not in self.index2idx:
            self.index2idx[index] = len(self.index2idx)
        return self.index2idx[index]        

def encode_condition_op(condition_op, encoding, ds_info):
    constants = ds_info.constants
    condition_op_dim, bool_ops_total_num, column_total_num, compare_ops_total_num = \
        constants.condition_op_dim, constants.bool_ops_total_num, \
        constants.column_total_num, constants.compare_ops_total_num
    
    # bool_operator + left_value + compare_operator + right_value
    if (condition_op is None) or (condition_op=='NA'):
        vec = [0 for _ in range(condition_op_dim)]
    elif isinstance(condition_op,str): # join
        l, op, r = condition_op.split(' ')
        left_value_idx = encoding.encode_col(l)
        left_value_vec = [0 for _ in range(column_total_num)]
        left_value_vec[left_value_idx] = 1
        operator_idx = encoding.encode_op(op)
        operator_vec = [0 for _ in range(compare_ops_total_num)]
        operator_vec[operator_idx] = 1
        right_value_idx = encoding.encode_col(r)
        right_value_vec = [0]
        left_value_vec[right_value_idx] = 1
        vec = [0 for _ in range(bool_ops_total_num)]
        vec = vec + left_value_vec + operator_vec + right_value_vec
    else:
        col = condition_op[0]
        left_value_idx = encoding.encode_col(col)
        left_value_vec = [0 for _ in range(column_total_num)]
        left_value_vec[left_value_idx] = 1
        right_value = condition_op[2]

        operator_idx = encoding.encode_op(condition_op[1])
        operator_vec = [0 for _ in range(compare_ops_total_num)]
        operator_vec[operator_idx] = 1
        
        right_value_vec = [encoding.normalize_val(col, right_value)]

        vec = [0 for _ in range(bool_ops_total_num)]
        vec = vec + left_value_vec + operator_vec + right_value_vec
    num_pad = condition_op_dim - len(vec)
    result = np.pad(vec, (0, num_pad), 'constant')
#     print 'condition op: ', result
    return result

def encode_condition(conditions, ds_info, encoding):
    if conditions is None or len(conditions) == 0:
        vecs = [[0 for _ in range(ds_info.constants.condition_op_dim)]]
    else:
        vecs = [encode_condition_op(condition, encoding, ds_info) for condition in conditions] 
    num_pad = ds_info.constants.condition_max_num - len(vecs)
    result = np.pad(vecs, ((0, num_pad),(0,0)), 'constant')
    return result


class Constants:
    def __init__(self, ds_info):
        self.column_total_num = len(ds_info.columns) + 5
        self.extra_info_num = max(len(ds_info.columns), len(ds_info.tables), len(ds_info.indexes)) + 5
        self.operator_len = len(ds_info.nodeTypes) + 5
        self.condition_max_num = ds_info.max_filters + 5
        self.bool_ops_total_num = 5
        self.compare_ops_total_num = 15
        self.condition_op_dim = self.bool_ops_total_num + \
            self.compare_ops_total_num + self.column_total_num + 1 #0 # +1000 (sample)


## to add max_condition_num
def encode_node(node, ds_info, encoding): 
    column_total_num = ds_info.constants.column_total_num
    extra_info_num = ds_info.constants.extra_info_num
    operator_vec = np.zeros(ds_info.constants.operator_len)
    extra_info_vec = np.zeros(extra_info_num)
    
    condition_max_num = ds_info.constants.condition_max_num
    bool_ops_total_num = ds_info.constants.bool_ops_total_num
    compare_ops_total_num = ds_info.constants.compare_ops_total_num
    condition_op_dim = ds_info.constants.condition_op_dim
    condition_vec = np.zeros((condition_max_num, condition_op_dim))
    
    has_condition = 0
    if node is not None:
        operator = node.nodeType
        operator_idx = encoding.encode_type(operator)
        operator_vec[operator_idx] = 1
        if operator == 'Materialize' or operator == 'BitmapAnd' or operator == 'Result':
            pass
        elif operator in ['Hash Join', 'Merge Join','Nested Loop']:
            condition_vec = encode_condition(node.filters + [node.join], ds_info, encoding)
            has_condition = 1
#         elif operator in ['Seq Scan', 'Bitmap Heap Scan', 'Index Scan', 'Bitmap Index Scan', 'Index Only Scan']:
        else:
            relation_name = node.table
            index_name = node.index
            if relation_name != None:
                extra_info_inx = encoding.encode_table(relation_name)
            else:
                extra_info_inx = encoding.encode_index(index_name)
            extra_info_vec[extra_info_inx] = 1
            condition_vec = encode_condition(node.filters, ds_info, encoding)
            has_condition = 1
    return operator_vec, extra_info_vec, condition_vec, has_condition


def encode_plan(plan, ds_info, encoding):
    operators, extra_infos, conditions, condition_masks = [], [], [], []
    mapping = []
    
#     nodes_by_level = []
#     node = TreeNode(plan[0], None, 0, -1) # root indeed
#     recover_tree(plan[1:], node, 1) # from seq (from json dfs to seq) back to node plan 
#    dfs_tree_to_level(node, 0, nodes_by_level)  # re-written
    def dfs_tree(plan):
        nodes_by_level = []
#         idx = 0
        def dfs(node, lvl = 0):
#             nonlocal idx
            if len(nodes_by_level) <= lvl:
                nodes_by_level.append([])
            nodes_by_level[lvl].append(node)
            node.idx = len(nodes_by_level[lvl])#-1
#             idx += 1
            if node.children is not None:
                for child in node.children:
                    dfs(child, lvl+1)
        dfs(plan,0)
        return nodes_by_level

    nodes_by_level = dfs_tree(plan)
        
    ## tmp
#     plan.idx, plan.children[0].idx, plan.children[0].children[0].idx, \
#         plan.children[0].children[1].idx, plan.children[0].children[1].children[0].idx = (0, 1, 2, 4, 5)
    
    for level in nodes_by_level:
        operators.append([])
        extra_infos.append([])
        conditions.append([])
#         samples.append([])
        condition_masks.append([])
        mapping.append([])
        for node in level:
            operator, extra_info, condition, condition_mask = encode_node(node, ds_info, encoding)
            operators[-1].append(operator)
            extra_infos[-1].append(extra_info)
            conditions[-1].append(condition)
#             samples[-1].append(sample)
            condition_masks[-1].append(condition_mask)
            if len(node.children) == 2:
                mapping[-1].append([n.idx for n in node.children])
            elif len(node.children) == 1:
                mapping[-1].append([node.children[0].idx, 0])
            else:
                mapping[-1].append([0, 0])

    return operators, extra_infos, conditions, condition_masks, mapping


class E2E_Dataset(Dataset):
    def __init__(self, nodes, labels, encoding, ds_info, max_filters=10, max_node=20, rel_pos_max=10):

        self.encoding = encoding

        self.max_filters = max_filters
        self.max_node = max_node 
        self.rel_pos_max = rel_pos_max
        
        self.length = len(nodes)
        self.nodes = nodes
        self.ds_info = ds_info
        
        self.costs = labels
        self.cost_labels = ds_info.cost_norm.normalize_labels(labels)

        ## operators, extra_infos, conditions, condition_masks, mapping
        self.features = [encode_plan(node, ds_info, encoding) for node in nodes]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        return self.features[idx], self.cost_labels[idx]
      

class Batch():
    def __init__(self, operators, extra_infos, conditions, condition_masks, mapping, y):
        super(Batch, self).__init__()
        self.operators = operators
        self.extra_infos, self.y = extra_infos, y
        self.conditions = conditions
        self.condition_masks = condition_masks
        self.mapping = mapping
    def to(self, device):
        self.operators = self.operators.to(device)
        self.y = self.y.to(device)
        self.extra_infos = self.extra_infos.to(device)
        self.conditions, self.condition_masks = self.conditions.to(device), self.condition_masks.to(device)
        self.mapping = self.mapping.to(device)
        return self

    def __len__(self):
        return self.y.size(0)


def collator(small_set):
#     collator.counter += 1
#     print(collator.counter)
    
    y = [s[1] for s in small_set]
#     print(y)
    xs =  [s[0] for s in small_set]
    operators = [s[0] for s in xs]
    extra_infos = [s[1] for s in xs]
    conditions = [s[2] for s in xs]
    condition_masks = [s[3] for s in xs]
    mapping = [s[4] for s in xs]

    operators_batch = []
    extra_infos_batch = []
    conditions_batch = []
    condition_masks_batch = []
    mapping_batch = []
    for i in range(len(operators)):
        operators_batch = merge_plans_level(operators_batch, operators[i])
        extra_infos_batch = merge_plans_level(extra_infos_batch, extra_infos[i])
        conditions_batch = merge_plans_level(conditions_batch, conditions[i])
        condition_masks_batch = merge_plans_level(condition_masks_batch, condition_masks[i])
        mapping_batch = merge_plans_level(mapping_batch, mapping[i], True)
    max_nodes = 0
    for o in operators_batch:
        if len(o) > max_nodes:
            max_nodes = len(o)
#     print (max_nodes)
#     print (len(condition2s_batch))
    operators_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in operators_batch])
    extra_infos_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in extra_infos_batch])
    conditions_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0),(0,0)), 'constant') for v in conditions_batch])
    condition_masks_batch = np.array([np.pad(v, (0, max_nodes - len(v)), 'constant') for v in condition_masks_batch])
    mapping_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in mapping_batch])
    
#     print ('operators_batch: ', operators_batch.shape)
    operators_batch = torch.FloatTensor([operators_batch]).squeeze(0)
    extra_infos_batch = torch.FloatTensor([extra_infos_batch]).squeeze(0)
    conditions_batch = torch.FloatTensor([conditions_batch]).squeeze(0)
    condition_masks_batch = torch.FloatTensor([condition_masks_batch]).squeeze(0)
    mapping_batch = torch.FloatTensor([mapping_batch]).squeeze(0)
    
    y = torch.FloatTensor(y).unsqueeze(1)
    return Batch(operators_batch, extra_infos_batch, conditions_batch, condition_masks_batch, mapping_batch,y), y
  

import copy
def merge_plans_level(level1, level2, isMapping=False):
    l = copy.deepcopy(level1)
    l2 = copy.deepcopy(level2)
    for idx in range(len(l2)):
        if idx >= len(l):
            l.append([])
        if isMapping:
            if len(l) > idx+1:
                base = len(l[idx+1])
            else:
                base = 0
            for i in range(len(l2[idx])):
                if l2[idx][i][0] > 0:
                    l2[idx][i][0] += base
                if l2[idx][i][1] > 0:
                    l2[idx][i][1] += base
        l[idx] += l2[idx]
    return l




