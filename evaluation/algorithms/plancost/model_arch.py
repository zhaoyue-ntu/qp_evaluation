import torch
import torch.nn as nn
from torch.nn import init

from ..queryformer.model import Prediction
import sys
sys.path.append('../evaluation/')
from algorithms.plancost import featurize
import functools, os
import numpy as np
import json
from copy import deepcopy



# +
"""
                       Operator Neural Unit Architecture                    #
##############################################################################
Neural Unit that covers all operators
"""
class NeuralUnit(nn.Module):
    """Define a Resnet block"""

    def __init__(self, node_type, dim_dict, num_layers=5, hidden_size=128,
                 output_size=32, norm_enabled=False):
        """
        Initialize the InternalUnit
        """
        super(NeuralUnit, self).__init__()
        self.node_type = node_type
        self.dense_block = self.build_block(num_layers, hidden_size, output_size,
                                            input_dim = dim_dict[node_type])

    def build_block(self, num_layers, hidden_size, output_size, input_dim):
        """Construct a block consisting of linear Dense layers.
        Parameters:
            num_layers  (int)
            hidden_size (int)           -- the number of channels in the conv layer.
            output_size (int)           -- size of the output layer
            input_dim   (int)           -- input size, depends on each node_type
            norm_layer                  -- normalization layer
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        assert(num_layers >= 2)
#         dense_block = nn.ModuleList([nn.Linear(input_dim, output_size), nn.ReLU()])
        
        dense_block = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
#         dense_block = nn.ModuleList([nn.Linear(input_dim, hidden_size), nn.ReLU()])
#         for i in range(num_layers - 2):
#             dense_block.append(nn.Linear(hidden_size, hidden_size))
#             dense_block.append(nn.ReLU())
#         dense_block.append(nn.Linear(hidden_size, output_size))

#         for layer in dense_block:
#             try:
#                 nn.init.xavier_uniform_(layer.weight)
#             except:
#                 pass
            
        # prediction = Prediction(32)
        # return nn.Sequential(*dense_block, prediction)
#         return nn.Sequential(*dense_block)
        return dense_block

    def forward(self, x):
        """ Forward function """
        out = self.dense_block(x)
        return out
# -





class QPPNet(nn.Module):
    def __init__(self, args, REL_NAMES, REL_ATTR_LIST_DICT, index_names, col_min_max, ds_info):
        super(QPPNet, self).__init__()
        self.__cuda = False
        self.batch_size = args.bs
        # self.device = args.device
        self.units = {}
        self.num_rel = len(REL_NAMES)
        self.max_num_attr = max([len(i) for i in REL_ATTR_LIST_DICT.values()])
        self.num_index = len(index_names)

        self.dim_dict = {'Seq Scan': self.num_rel + self.max_num_attr * 2 + 3,
                    'Index Scan': self.num_index + self.num_rel + self.max_num_attr * 2 + 3 + 1,
                    'Index Only Scan': self.num_index + self.num_rel + self.max_num_attr * 2 + 3 + 1,
                    'Bitmap Heap Scan': self.num_rel + self.max_num_attr * 2 + 3,
                    'Bitmap Index Scan': self.num_rel + self.max_num_attr * 2 + 3,
                    'Sort': 3 + 2 + self.num_rel * self.max_num_attr + 32,
                    'Hash': 4 + 32,             
                    'Hash Join': 10 + 3 + 32 * 2,
                    'Merge Join': 10 + 3 + 32 * 2,
                    'Aggregate': 3 + 4 + 1 + 32, 'Nested Loop': 32 * 2 + 3, 'Limit': 32 + 3,
                    'Subquery Scan': 32 + 3,
                    'Materialize': 32 + 3, 'Gather Merge': 32 + 3, 'Gather': 32 + 3,
                    'WindowAgg': 3 + 32,
                    'Append': 3 + 32,
                    'CTE Scan': self.num_rel + self.max_num_attr * 2 + 3,
                    'Incremental Sort': 3 + 2 + self.num_rel * self.max_num_attr + 32,
                    'Merge Append': 128 + 5 + 32,
                    'Group': 32 + 3,
                    'SetOp': 32 + 3 + 2,
                    'Unique': 32 + 3,
                    'Result': 32 + 3,
                    'BitmapAnd': 32 + 3,
                    'BitmapOr': 32 + 3}

        self.f = featurize.featurizer(REL_NAMES, REL_ATTR_LIST_DICT, index_names, col_min_max, ds_info)
        
        self.unit_list = nn.ModuleList()
        
        
        for ind, operator in enumerate(self.dim_dict):
            self.unit_list.append(NeuralUnit(operator, self.dim_dict))
            self.units[operator] = self.unit_list[-1]
            
    def device(self):
        return next(self.parameters()).device

    def get_input(self, data): 
        """
            Vectorize the input of a list of queries that have the same plan structure (of the same template/group)

            Args:
            - data: a list of plan_dict, each plan_dict correspond to a query plan in the dataset;
                    requires that all plan_dicts is of the same query template/group

            Returns:
            -- new_samp_dict: a dictionary, where each level has the following attribute:
            -- node_type     : name of the operator
            -- subbatch_size : number of queries in data
            -- feat_vec      : a numpy array of shape (batch_size x feat_dim) that's
                               the vectorized inputs for all queries in data
            -- children_plan : list of dictionaries with each being an output of
                               a recursive call to get_input on a child of current node
            -- total_time    : a vector of prediction target for each query in data
            -- is_subplan    : if the queries are subplans
        """

        new_samp_dict = {}
        new_samp_dict["node_type"] = data[0].nodeType
        new_samp_dict["subbatch_size"] = len(data)
#         for root in data:
#             print(root, self.f.featurize(root))
        feat_vec = np.array([np.nan_to_num(self.f.featurize(root), 0) for root in data])
        

        total_time = [root.cost for root in data]
        child_plan_lst = []
        if data[0].children != []:
#             print(data[0].children)
            for i in range(len(data[0].children)):
                child_plan_dict = self.get_input([root.children[i] for root in data])
                child_plan_lst.append(child_plan_dict)
#                 print(child_plan_dict)
                
        new_samp_dict["feat_vec"] = np.array(feat_vec).astype(np.float32)
        new_samp_dict["children_plan"] = child_plan_lst
        new_samp_dict["total_time"] = np.array(total_time).astype(np.float32)

        if data[0].subplan_name:
            new_samp_dict['is_subplan'] = True
        else:
            new_samp_dict['is_subplan'] = False
        return new_samp_dict


    def construct_tree_net(self, samp_batch):
        feat_vec = samp_batch['feat_vec']
        input_vec = torch.from_numpy(feat_vec).to(self.device())
        
        # subplans_time = []
        for child_plan_dict in samp_batch['children_plan']:
#             print(child_plan_dict['feat_vec'])
            child_output_vec = self.construct_tree_net(child_plan_dict)
            input_vec = torch.cat((input_vec, child_output_vec),axis=1)
            
        expected_len = self.dim_dict[samp_batch['node_type']]
        if expected_len > input_vec.size()[1]:
            add_on = torch.zeros(input_vec.size()[0], expected_len - input_vec.size()[1])
        # commented .to
        #             add_on = add_on.to(input_vec.device)
            add_on = add_on.to(input_vec.device)
        #
            input_vec = torch.cat((input_vec, add_on), axis=1)

        if expected_len < input_vec.size()[1]:
            input_vec = input_vec[:, :expected_len]
        input_vec = torch.nan_to_num(input_vec, 0)
#         print('input_vec', input_vec)
        output_vec = self.units[samp_batch['node_type']](input_vec)
#         print('output_vec', output_vec)
        return output_vec#, pred_time
    
            
    def forward(self, X):        
        
        output_vec = self.construct_tree_net(self.get_input(X.ins))

        return output_vec

            






