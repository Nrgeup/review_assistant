# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# from torch.nn.parallel import DistributedDataParallel
# from apex.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable

import numpy as np
import os
import pickle
import time
import random
import secrets
import math
import pandas as pd

# from transformers import *
from data_setup import get_cuda, ModelHelper, Tokenizer4Bert, pad_and_truncate


# from easynlp.data_utils.data_setup import get_cuda
# from easynlp.data_utils.data_setup import ModelHelper, Tokenizer, Tokenizer4Bert
# from easynlp.data_utils.data_setup import load_word_vec
# from easynlp.data_utils.data_setup import pad_and_truncate

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics

from my_models import Model_sentiment

from transformers import BertModel

# from my_models import My_model10, My_model11, My_model12
import random


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, fname, tokenizer):
        # ['Id' 'ProductId' 'UserId' 'ProfileName' 'HelpfulnessNumerator'
        # 'HelpfulnessDenominator' 'Score' 'Time' 'Summary' 'Text' 'Sentiment'
        # 'Usefulness']
        df = pd.read_csv(fname, encoding='utf-8')
        all_data = []
        print(df.shape)

        for i in range(len(df)):
            row = df.iloc[i]
            if (i + 1) % 100000 == 0:
                print(i + 1)
            # print(i)
            text_raw = row['Text']
            text_summary = row['Summary']

            sentiment_polarity = 0
            if row['Sentiment'] == 'positive':
                sentiment_polarity = 1
            sentiment_score = float(row['Score']) / 5.0

            text_raw_indices = tokenizer.text_to_sequence('[CLS] ' + text_raw + ' [SEP]')

            data = {
                'text_raw_indices': text_raw_indices,
                'sentiment_polarity': sentiment_polarity,
                'sentiment_score': sentiment_score,
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyModelHelper(ModelHelper):
    def __init__(self, opt):
        self.opt = opt
        self.opt.logger.info("Start initialization...")

        # Step 1: Tokenizer
        self.tokenizer = Tokenizer4Bert(self.opt.max_seq_len, self.opt.pretrained_bert_name)
        self.opt.logger.info("End of preparing tokenizer...")

        # Step 2: Dataset
        if self.opt.if_train:
            self.opt.train_data_file = self.opt.dataset_path + "train_data.csv"
            # self.opt.train_data_file = self.opt.dataset_path + "test_data.csv"  # Only for faster debugging
            self.train_dataset = MyDataset(self.opt.train_data_file, self.tokenizer)
            self.opt.logger.info("End of preparing training dataset...")
        self.opt.test_data_file = self.opt.dataset_path + "test_data.csv"
        self.opt.train_batch_size = self.opt.batch_size
        self.test_dataset = MyDataset(self.opt.test_data_file, self.tokenizer)
        # self.test_dataset = self.train_dataset  # Only for faster debugging
        self.opt.logger.info("End of preparing test dataset...")

        # Step3: Model
        if self.opt.model == "sentiment-predict":
            base_model = BertModel.from_pretrained(self.opt.pretrained_bert_name)
            self.model = Model_sentiment(
                base_model,
                self.opt.dropout,
                self.opt.bert_dim,
                self.opt.polarities_dim
            ).to(self.opt.device)
            self.cols = ['text_raw_indices']
        self._reset_params()

        # if self.opt.n_gpu > 1:
        #     self.model = DistributedDataParallel(
        #         self.model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank,
        #         find_unused_parameters=True
        #     )

        if self.opt.if_load_from_checkpoint:
            self.model.load_state_dict(torch.load(self.opt.model_save_path + self.opt.model_file))

        self.opt.logger.info("End of creating models...")
        if self.opt.local_rank < 1:
            self.print_args()

    def save_model(self):
        torch.save(self.model.state_dict(), self.opt.model_save_path + self.opt.model_file)

    def _reset_params(self):
        for child in self.model.children():
            # if type(child) == nn.Linear:  # Third
            #     child.weight.data.normal_(mean=0.0, std=0.02)

            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p) # First
                            # nn.init.xavier_normal_(p) # Second

    def configure_loss(self):
        # Compute loss
        if self.opt.polarities_dim == 1:
            #  We are doing regression
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.NLLLoss()
        return criterion

    def run(self):
        if self.opt.if_train:
            self.train()
        else:
            acc, f1 = self.test()
            print("test_acc: {:.4f} test_f1: {:.4f}".format(acc, f1))
        input("Done!")

    def train(self):
        self.opt.logger.info("Start training...")
        # Dataloader
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.opt.train_batch_size)
        self.opt.logger.info("End of data loader...")
        # Optimizer
        # optimizer = AdamW(self.model.parameters(), lr=self.opt.lr, correct_bias=False)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.opt.num_warmup_steps,
                                                    num_training_steps=self.opt.num_total_steps)  # PyTorch scheduler
        self.opt.logger.info("End of building optimizer")

        criterion = self.configure_loss()

        max_test_acc = 0
        max_test_f1 = 0
        tot_time = 0
        self.model.zero_grad()
        step = 1
        for epoch in range(1, self.opt.num_epoch + 1):
            start_cpu_secs = time.time()
            # train_sampler.set_epoch(epoch)
            n_correct, n_total, loss_total = 0, 0, 0.0
            for i_batch, sample_batched in enumerate(train_dataloader):
                # switch model to training mode
                self.model.train()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.cols]
                targets = sample_batched['sentiment_polarity'].to(self.opt.device)
                model_outputs = self.model(inputs)

                loss = criterion(model_outputs[0], targets)

                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                model_predict = model_outputs[0]

                n_correct += (torch.argmax(model_predict, -1) == targets).sum().item()
                n_total += len(model_predict)
                loss_total += loss.item() * len(model_predict)

                if i_batch % 20 == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    if self.opt.local_rank < 1:
                        self.opt.logger.info('Train: Epoch: {}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
                            epoch, i_batch, train_loss, train_acc))
                    n_correct, n_total, loss_total = 0, 0, 0.0

                step += 1

                if i_batch % 200 == 0:
                    end_cpu_secs = time.time()
                    test_acc, test_f1 = self.test()

                    self.opt.logger.info('Test: Epoch: {} of {} took: {:.3f}s, test_acc: {:.4f}(best: {:.4f}), test_f1: {:.4f}(best: {:.4f})'.format(
                        epoch, self.opt.num_epoch, end_cpu_secs - start_cpu_secs, test_acc, max_test_acc, test_f1, max_test_f1
                    ))
                    tot_time += end_cpu_secs - start_cpu_secs
                    self.opt.logger.info("Done! Total time= {:.3f}s".format(tot_time))
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        self.save_model()
                        self.opt.logger.info("Create highest acc: {}".format(max_test_acc))
                    # if test_f1 > max_test_f1:
                        max_test_f1 = test_f1

    def test(self):
        self.opt.logger.info("Start testing...")
        # test_sampler = RandomSampler(self.test_dataset) if self.opt.n_gpu > 1 else DistributedSampler(
        #     self.train_dataset)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.opt.train_batch_size, shuffle=False)

        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all, b_outputs_all = None, None, None
        # switch model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_dataloader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.cols]
                t_targets = t_sample_batched['sentiment_polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)[0]
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        print("n_total", n_total)
        return acc, f1




