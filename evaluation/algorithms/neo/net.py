import torch.nn as nn
import torch
from torch import tensor, FloatTensor, cat
import numpy as np
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
import time

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

def append_tensor_to_tree(tree, tensor):
    if tree.dim() == 1:
        # If the current node is a tensor, append the new tensor to it
        return torch.cat([tree, tensor], dim=0)
    else:
        # If the current node is a tree, recursively append tensor to each node
        return [append_tensor_to_tree(child, tensor) for child in tree]



class NeoNet(nn.Module):
    def __init__(self, in_channels, rel_names, rel_attr_list_dict):
        super(NeoNet, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 512),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            DynamicPooling(),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, 1)
        )
        
        self._relations = rel_names
        self._attr_dict = rel_attr_list_dict    
        self.NUM_REL = len(self._relations)
        self.len_qvec = sum([x for x in range(self.NUM_REL)]) + sum([len(x) for x in self._attr_dict.values()])
        self.q_conv = nn.Sequential(nn.Linear(self.len_qvec, 64),
                           nn.Linear(64, 128),
                           nn.Linear(128, 64),
                           nn.Linear(64, 32))

    def in_channels(self):
        return self.__in_channels
    
            
    def forward(self, x):
#         start_time_q_mlp = time.time()
#         q_vecs = [self.q_conv(z) for z in x.q_vecs]
        q_vecs = self.q_conv(x.q_vecs)
#         print("--- q vec MLP takes %s seconds ---" % (time.time() - start_time_q_mlp))
        flat_trees = x.trees
        #concate here
#         start_time_concat = time.time()
        assert len(flat_trees) == len(q_vecs)
    
        B, N = flat_trees.size()[:2]
        q_vecs_repeat = q_vecs.view(B,1,-1).repeat(1,N,1)
        aug_trees = torch.cat([flat_trees, q_vecs_repeat],dim=2)
#         aug_trees = FloatTensor([[[0]*(len(flat_trees[0][0])+len(q_vecs[0]))]*len(flat_trees[0])]*len(flat_trees)).to(self.device)
#         for i in range(len(flat_trees)):
#             for j in range(len(flat_trees[i])):
#                 tr = torch.cat((flat_trees[i][j], q_vecs[i]))
#                 aug_trees[i][j] = tr
#         aug_trees = [append_tensor_to_tree(flat_trees[i], q_vecs[i]) for i in range(len(q_vecs))]
#         print("--- concate aug_tree takes %s seconds ---" % (time.time() - start_time_concat))



        aug_trees = aug_trees.transpose(1, 2)
#         start_time_conv = time.time()
        conv = self.tree_conv((aug_trees, x.idxes))
#         print("--- tree convolution takes %s seconds ---" % (time.time() - start_time_conv))
        return conv

    def cuda(self):
        self.__cuda = True
        return super().cuda()
