# -*- coding: utf-8 -*-

######################################################################################
#  Import Packages
######################################################################################
# Basic Packages
import os
import time
import argparse
import math
import numpy
import torch
import torch.nn as nn
import ast
import numpy as np
from code_useful.utils import split_sentence
from code_useful.data_setup import Tokenizer4Bert

######################################################################################
#  End of hyper parameter
######################################################################################
pretrained_bert_name = "bert-base-uncased"
max_seq_len = 80
max_sent_len = 6


def output_useful(self_model, input_text):
    if input_text is None:
        input_text = "Hello, my dog is cute"

    my_tokenizer = Tokenizer4Bert(80, pretrained_bert_name)
    input_sents = split_sentence(input_text)
    input_sents[0] = '[CLS] ' + input_sents[0]
    input_sents[-1] = input_sents[-1] + ' [SEP]'
    sent_raw_indices, masks = my_tokenizer.sent_to_sequence(input_sents)
    sent_masks = []
    for item in masks:
        sent_masks.append(item[:max_seq_len])
    while len(sent_masks) < max_sent_len:
        sent_masks.append([0] * max_seq_len)
    sent_masks = np.array(sent_masks, dtype='float')
    sent_lens = np.array(
        [1] * len(input_sents) + [0] * (max_sent_len - len(input_sents))
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inputs = [
        torch.tensor(sent_raw_indices).unsqueeze(0).to(device),
        torch.tensor(sent_masks).unsqueeze(0).to(device),
        torch.tensor(sent_lens).unsqueeze(0).to(device),
    ]
    model_outputs = self_model(inputs)

    sentence_outputs = model_outputs[1]
    overall_outputs = model_outputs[0]

    overall_useful = torch.nn.functional.softmax(overall_outputs, 1)
    if overall_useful.data[0][0] > overall_useful.data[0][1]:
        predict_label = 0
    else:
        predict_label = 1

    sentence_useful = torch.nn.functional.softmax(sentence_outputs, 2)
    sentence_data = sentence_useful.data[0].cpu().numpy()
    sentence_info = []
    for index, item in enumerate(input_sents):
        sentence_score = sentence_data[index]
        sentence_info.append([item, sentence_score[0], sentence_score[1]])

    return predict_label, sentence_info














