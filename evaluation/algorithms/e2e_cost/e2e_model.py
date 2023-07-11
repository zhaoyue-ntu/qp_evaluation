
import torch
import torch.nn.functional as F
import torch.nn as nn

import time


class E2E_model(nn.Module):
    def __init__(self, hidden_dim, hid_dim, middle_result_dim, ds_info):
        super(E2E_model, self).__init__()
        constants = ds_info.constants
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(constants.condition_op_dim, hidden_dim, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(hid_dim)
        # The linear layer that maps from hidden state space to tag space
        
        self.condition_mlp = nn.Linear(hidden_dim, hid_dim)

        
#         self.lstm2 = nn.LSTM(15+108+2*hid_dim, hidden_dim, batch_first=True) #这么hardcode不怕你妈没了？
        one_hot_dim = constants.operator_len + constants.extra_info_num
    
        self.lstm2 = nn.LSTM(one_hot_dim+hid_dim, hidden_dim, batch_first=True)
        

        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hid_mlp2_task1 = nn.Linear(hidden_dim, hid_dim)
        self.batch_norm3 = nn.BatchNorm1d(hid_dim)
        self.hid_mlp3_task1 = nn.Linear(hid_dim, hid_dim)
        self.out_mlp2_task1 = nn.Linear(hid_dim, 1)

    #         self.hidden2values2 = nn.Linear(hidden_dim, action_num)

    def init_hidden(self, hidden_dim, batch_size=1, device = 'cpu'):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, hidden_dim).to(device),
                torch.zeros(1, batch_size, hidden_dim).to(device))
    
    def forward(self, batch):
        operators, extra_infos, conditions, condition_masks, mapping = \
            batch.operators, batch.extra_infos, batch.conditions, batch.condition_masks, batch.mapping
        # condition1
        batch_size = 0
        for i in range(operators.size()[1]):
            if operators[0][i].sum(0) != 0:
                batch_size += 1
            else:
                break
#         print ('batch_size: ', batch_size)
        
        num_level = conditions.size()[0]
        num_node_per_level = conditions.size()[1]
        num_condition_per_node = conditions.size()[2]
        condition_op_length = conditions.size()[3]
        
        inputs = conditions.view(num_level * num_node_per_level, num_condition_per_node, condition_op_length)
        
        device = operators.device
        hidden = self.init_hidden(self.hidden_dim, num_level * num_node_per_level, device = device)
        
        out, hid = self.lstm1(inputs, hidden)
        last_output1 = hid[0].view(num_level * num_node_per_level, -1)
        
        
        last_output1 = F.relu(self.condition_mlp(last_output1))
        last_output = last_output1
        last_output = self.batch_norm1(last_output).view(num_level, num_node_per_level, -1)
        
#         print (last_output.size())
#         torch.Size([14, 133, 256])
        
#         sample_output = F.relu(self.sample_mlp(samples))
#         sample_output = sample_output * condition_masks

#         out = torch.cat((operators, extra_infos, last_output, sample_output), 2)

        out = torch.cat((operators, extra_infos, last_output), 2) # minus a hid_dim

#         torch.Size([14, 133, 635])
#         out = out * node_masks

        hidden = self.init_hidden(self.hidden_dim, num_node_per_level, device = device)

        last_level = out[num_level-1].view(num_node_per_level, 1, -1)
#         torch.Size([133, 1, 635])
        _, (hid, cid) = self.lstm2(last_level, hidden)
        
        mapping = mapping.long()
        for idx in reversed(range(0, num_level-1)):
            mapp_left = mapping[idx][:,0]
            mapp_right = mapping[idx][:,1]
            pad = torch.zeros_like(hid)[:,0].unsqueeze(1)
            next_hid = torch.cat((pad, hid), 1)
            pad = torch.zeros_like(cid)[:,0].unsqueeze(1)
            next_cid = torch.cat((pad, cid), 1)
#             print(next_hid.shape, mapp_left, mapp_right)
            hid_left = torch.index_select(next_hid, 1, mapp_left)
            cid_left = torch.index_select(next_cid, 1, mapp_left)
            hid_right = torch.index_select(next_hid, 1, mapp_right)
            cid_right = torch.index_select(next_cid, 1, mapp_right)
            hid = (hid_left + hid_right) / 2
            cid = (cid_left + cid_right) / 2
            last_level = out[idx].view(num_node_per_level, 1, -1)
            _, (hid, cid) = self.lstm2(last_level, (hid, cid))
        output = hid[0]
#         print (output.size())
#         torch.Size([133, 128])

        last_output = output[0:batch_size]
        out = self.batch_norm2(last_output)
        
        return out
        
