import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertForSequenceClassification
from transformers.modeling_bert import BertPooler, BertSelfAttention
import copy
import math
import random
from torch.autograd import Variable


class Model_sentiment(nn.Module):
    def __init__(self, base_LM, dropout, LM_dim, polarities_dim):
        super(Model_sentiment, self).__init__()
        self.base_LM = base_LM
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(LM_dim, LM_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(LM_dim, polarities_dim)
        return

    def forward(self, inputs=None):
        text_raw_indices = inputs[0]

        # Pre-training LM
        representation = self.base_LM(text_raw_indices)[1]  # last_hidden_state, pooler_output
        representation = self.dropout(representation)

        # Project
        out = self.fc1(representation)
        out = self.relu(out)
        out = self.fc2(out)

        return [out]
