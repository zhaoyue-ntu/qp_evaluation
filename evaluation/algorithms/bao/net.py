import torch.nn as nn
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
# from TreeConvolution.util import prepare_trees

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

class BaoNet(nn.Module):
    def __init__(self, in_channels):
        super(BaoNet, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling()
#             nn.Linear(64, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, 1)
        )

    def in_channels(self):
        return self.__in_channels
        
    def forward(self, x):
        # trees = prepare_trees(x, features, left_child, right_child,
        #                       cuda=self.__cuda)
        return self.tree_conv((x.trees, x.idxes))

    def cuda(self):
        self.__cuda = True
        return super().cuda()
