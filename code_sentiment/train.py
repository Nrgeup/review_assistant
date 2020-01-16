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

from enviroment_setup import set_logger, set_gpu, set_seed

from model_helper import MyModelHelper

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
parser.add_argument('--if_train', type=ast.literal_eval, default=False, help='If train model.')
parser.add_argument('--if_load_from_checkpoint', type=ast.literal_eval, default=False, help='If load from saved checkpoint.')
parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", default=-1, type=int, help="For distributed environment.")

# -----------------  File parameters  ----------------- #
parser.add_argument('--checkpoint_name', type=str, default="None", help='Saved checkpoint name.')
parser.add_argument('--model_file', type=str, default="model_params.model", help='Saved model name.')
parser.add_argument('--dataset_path', type=str, default="../datasets/amazon-fine-food-reviews/")


# -----------------  Hyper parameters  ----------------- #
parser.add_argument('--model', default='sentiment-predict', type=str)
parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
parser.add_argument('--max_seq_len', default=80, type=int)
parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--dropout', default=None, type=float)
parser.add_argument('--bert_dim', default=None, type=int)
parser.add_argument('--polarities_dim', default=None, type=int)


parser.add_argument('--lr', default=None, type=float)
parser.add_argument('--max_grad_norm', default=None, type=float)
parser.add_argument('--num_total_steps', default=None, type=int)
parser.add_argument('--num_warmup_steps', default=None, type=int)

hparams = parser.parse_args()
######################################################################################
#  End of hyper parameters
######################################################################################


def pre_setup():
    # Enviroment setup
    set_logger(hparams)
    set_gpu(hparams)
    set_seed(hparams)
    hparams.logger.info("Enviroment setup success!")


if __name__ == '__main__':
    pre_setup()
    model_helper = MyModelHelper(hparams)
    model_helper.run()



