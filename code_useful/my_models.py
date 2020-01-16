import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertForSequenceClassification
from transformers.modeling_bert import BertPooler, BertSelfAttention
import copy
import math
import random
from torch.autograd import Variable


class Useful_predict(nn.Module):
    def __init__(self, base_LM, dropout, LM_dim, polarities_dim):
        super(Useful_predict, self).__init__()
        self.base_LM = base_LM
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(LM_dim, LM_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(LM_dim, polarities_dim)
        return

    def forward(self, inputs=None):
        text_raw_indices = inputs[0]
        sent_masks = inputs[1]
        sent_num = inputs[2]

        sent_dims = sent_masks.size()
        
        # Pre-training LM
        representation = self.base_LM(text_raw_indices)[0]  # last_hidden_state, pooler_output
        representation = self.dropout(representation)
        rep_dims = representation.size()
        
        sent_4d = sent_masks.view(sent_dims[0], sent_dims[1], sent_dims[2], 1)
        sent_4ds = sent_4d.repeat(1, 1, 1, rep_dims[-1])
        
        rep_4d = representation.view(rep_dims[0], 1, rep_dims[1], rep_dims[2])
        rep_4ds = rep_4d.repeat(1, sent_dims[1], 1, 1)
        
        masked_rep = sent_4ds * rep_4ds
        sent_reps = torch.sum(masked_rep, dim=-2)
        
        # Project for each sentence
        sent_out = self.fc1(sent_reps)
        sent_out = self.relu(sent_out)
        sent_out = self.fc2(sent_out)

        sent_num_dims = sent_num.size()
        sent_num_3d = sent_num.view(sent_num_dims[0], sent_num_dims[1], 1)
        sent_num_3ds = sent_num_3d.repeat(1, 1, sent_out.size()[-1])
        
        final_out = sent_out * sent_num_3ds

        # MIL: add each sentence
        out = torch.sum(final_out, dim=-2)
        return [out, final_out]
        
