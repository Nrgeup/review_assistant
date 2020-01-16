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

from transformers import GPT2Tokenizer, GPT2LMHeadModel

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


def generate(input_text, model, tok, device):
    predict_word = None
    input_ids = torch.tensor(tok.encode(input_text)).unsqueeze(0).to(device)
    logits = model(input_ids)[0][:, -1]  # (1, vocab_size)
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()  # (vocab_size)
    idx = torch.argmax(probs)
    # print("idx", idx)
    token = tok.convert_ids_to_tokens(int(idx))
    predict_word = tok.convert_tokens_to_string(token)
    return predict_word.strip()



def output_summary(input_text):
    if input_text is None:
        input_text = "Hello, my dog is cute"

    if hparams.model == "sentiment-summary":
        self_model = GPT2LMHeadModel.from_pretrained(hparams.pretrained_bert_name).to(hparams.device)
        self_model.load_state_dict(torch.load("./outputs/{}/{}".format(hparams.checkpoint_name, hparams.model_file)))
        tokenizer = GPT2Tokenizer.from_pretrained(hparams.pretrained_bert_name)

        flag = "Y"
        while flag!="N":
            predict_word = generate(input_text, self_model, tokenizer)
            input_text = input_text + ' ' + predict_word
            print(input_text)
            flag = input("Next ? Y/N")

        return []


def generation_pre_setup(checkpoint_name, device, pretrained_bert_name, model_file):
    generation_model = GPT2LMHeadModel.from_pretrained(pretrained_bert_name).to(device)
    generation_model.load_state_dict(torch.load("../code_generation/outputs/{}/{}".format(checkpoint_name, model_file)))
    generation_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_bert_name)
    return generation_model, generation_tokenizer



if __name__ == '__main__':
    pre_setup()
    while True:
        input_str = input("Input your text:")
        input_str.strip()
        output_summary(input_str)












