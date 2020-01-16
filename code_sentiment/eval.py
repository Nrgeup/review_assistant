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

from transformers import BertTokenizer, BertModel
from code_sentiment.my_models import Model_sentiment


######################################################################################
#  Parsing parameters
######################################################################################
parser = argparse.ArgumentParser(description="Amazon Fine Food Reviews Analysis system.")
'''parser.add_argument(
        "--task_name", 
        default=None, 
        type=str/int/float/bool, n_gpu
        required=True, 
        choices=[], 
        help="The name of the task to train.")
'''
# -----------------  Environmental parameters  ----------------- #

parser.add_argument('--checkpoint_name', type=str, default="1578973534", help='Saved checkpoint name.')
parser.add_argument('--model_file', type=str, default="model_params.model", help='Saved model name.')
parser.add_argument('--model', default='sentiment-predict', type=str)
parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)

# -----------------  Hyper parameters  ----------------- #
parser.add_argument('--max_seq_len', default=80, type=int)
parser.add_argument('--bert_dim', default=None, type=int)
parser.add_argument('--polarities_dim', default=None, type=int)
parser.add_argument('--input_text', type=str, default=None, help="Input text")



hparams = parser.parse_args()
######################################################################################
#  End of hyper parameters
######################################################################################


def pre_setup():
    # Enviroment setup
    hparams.timestamp = hparams.checkpoint_name
    if torch.cuda.is_available():
        hparams.device = torch.device("cuda")
    else:
        hparams.device = torch.device("cpu")


def output_sentiment(input_text):
    if input_text is None:
        input_text = "Hello, my dog is cute"

    if hparams.model == "sentiment-predict":
        base_model = BertModel.from_pretrained(hparams.pretrained_bert_name)
        self_model = Model_sentiment(
            base_model,
            0.0,
            hparams.bert_dim,
            hparams.polarities_dim
        ).to(hparams.device)

        self_model.load_state_dict(torch.load("./outputs/{}/{}".format(hparams.checkpoint_name, hparams.model_file)))

        embeddings_gradient = []
        def bh(m, gi, go):
            # print(m)
            # print("Grad Input")
            # print(gi)
            # print("Grad Output")
            # print(go)
            embeddings_gradient.append(go)

        self_model.base_LM.embeddings.register_backward_hook(bh)

        tokenizer = BertTokenizer.from_pretrained(hparams.pretrained_bert_name)

        token = tokenizer.tokenize('[CLS] ' + input_text + ' [SEP]')
        # ['[CLS]', 'hello', ',', 'my', 'dog', 'is', 'cute', '[SEP]']

        sequence = tokenizer.convert_tokens_to_ids(token)
        # [101, 7592, 1010, 2026, 3899, 2003, 10140, 102]

        # text_raw_indices = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
        #     0).to(hparams.device)  # Batch size 1

        text_raw_indices = torch.tensor(sequence).unsqueeze(
            0).to(hparams.device)

        inputs = [text_raw_indices]
        model_outputs = self_model(inputs)
        overall_sentiment = torch.nn.functional.softmax(model_outputs[0], 1)
        # print(overall_sentiment)  # tensor([[0.1263, 0.8737]], device='cuda:0', grad_fn=<SoftmaxBackward>)

        targets = torch.tensor([0]).to(hparams.device)
        predict_label = 1
        if overall_sentiment.data[0][0] > overall_sentiment.data[0][1]:
            targets = torch.tensor([1]).to(hparams.device)
            predict_label = 0

        loss = nn.CrossEntropyLoss()(model_outputs[0], targets)
        self_model.zero_grad()
        loss.backward()

        # print(embeddings_gradient[0][0])
        # print("-" * 10)
        # print(embeddings_gradient[0][0].cpu().numpy())
        # print("+" * 10)
        # print(numpy.sum(embeddings_gradient[0][0].cpu().numpy()[0], axis=1))
        # print(len(numpy.sum(embeddings_gradient[0][0].cpu().numpy()[0], axis=1)))

        important_score = numpy.sum(embeddings_gradient[0][0].cpu().numpy()[0], axis=1)

        if predict_label == 0:
            print("Negative")
        else:
            print("Positive")

        print(token[1:-1])
        print(important_score[1:-1])

        return [predict_label, ]


def sentiment_pre_setup(checkpoint_name, device, pretrained_bert_name, model_file):
    base_model = BertModel.from_pretrained(pretrained_bert_name)
    sentiment_model = Model_sentiment(
        base_model,
        0.0,
        768,
        2,
    ).to(device)

    sentiment_model.load_state_dict(torch.load("../code_sentiment/outputs/{}/{}".format(checkpoint_name, model_file)))
    sentiment_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
    return sentiment_model, sentiment_tokenizer




def analysis_sentiment(input_text, model, tok, device):
    embeddings_gradient = []
    def bh(m, gi, go):
        # print(m)
        # print("Grad Input")
        # print(gi)
        # print("Grad Output")
        # print(go)
        embeddings_gradient.append(go)
    model.base_LM.embeddings.register_backward_hook(bh)
    token = tok.tokenize('[CLS] ' + input_text + ' [SEP]')
    sequence = tok.convert_tokens_to_ids(token)
    text_raw_indices = torch.tensor(sequence).unsqueeze(0).to(device)
    inputs = [text_raw_indices]
    model_outputs = model(inputs)
    overall_sentiment = torch.nn.functional.softmax(model_outputs[0], 1)
    targets = torch.tensor([0]).to(device)
    predict_label = 1
    if overall_sentiment.data[0][0] > overall_sentiment.data[0][1]:
        targets = torch.tensor([1]).to(device)
        predict_label = 0
    loss = nn.CrossEntropyLoss()(model_outputs[0], targets)
    model.zero_grad()
    loss.backward()

    important_score = numpy.sum(embeddings_gradient[0][0].cpu().numpy()[0], axis=1)
    return [predict_label, token[1:-1], important_score[1:-1]]


if __name__ == '__main__':
    pre_setup()
    while True:
        input_str = input()
        input_str.strip()
        output_sentiment(input_str)












