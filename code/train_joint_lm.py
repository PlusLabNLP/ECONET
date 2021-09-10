# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from transformers import *
from new_models import RobertaJointLM, BertJointLM
from utils import load_data, select_field, cal_f1, flatten_answers, accuracy
from optimization import *
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                               Path.home() / '.pytorch_pretrained_roberta'))


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters                                                                                         
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--filename1",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file 1")
    parser.add_argument("--filename2",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file 2")
    ## Other parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--event_weight",
                        default=1.0,
                        type=float,
                        help="event mask task weight")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs                      
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))

    label_map_temporal = {k: v for
                          k, v in enumerate([x.strip() for x in open("%s/temporal_indicators.txt" % args.data_dir)])}

    with open("%s/all_vocab.json" % (args.data_dir)) as fp:
        vocab = json.load(fp)
    label_map_event = {v: k for k, v in enumerate(vocab.keys())}

    logger.info(len(label_map_temporal))
    logger.info(len(label_map_event))

    # temporal data
    if args.filename1 == "empty":
        with open("%s%s" % (args.data_dir, args.filename2)) as infile:
            train_features = json.load(infile)
    else:
        with open("%s%s" % (args.data_dir, args.filename1)) as infile:
            train_features = json.load(infile)

        # event data
        with open("%s%s" % (args.data_dir, args.filename2)) as infile:
            train_features += json.load(infile)

    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'mask_ids'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_offsets = torch.tensor(select_field(train_features, 'offset'), dtype=torch.long)
    all_labels = torch.tensor(select_field(train_features, 'label'), dtype=torch.long)
    all_tasks = torch.tensor(select_field(train_features, 'task'), dtype=torch.long)

    logger.info("id_size: {} mask_size: {}, segment_size: {}, offset_size: {}, offset_size: {}, label_size: {}".format(
        all_input_ids.size(), all_input_mask.size(), all_segment_ids.size(),
        all_offsets.size(), all_tasks.size(), all_labels.size()))

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_offsets, all_tasks, all_labels)

    # free memory
    del train_features
    del all_input_ids
    del all_input_mask
    del all_segment_ids
    del all_offsets
    del all_labels
    del all_tasks

    if "roberta" in args.model:
        if args.filename1 == "empty":
            model = RobertaJointLM.from_pretrained(args.model, mlp_hid=args.mlp_hid_size,
                                                   num_classes_tmp=len(label_map_temporal),
                                                   num_classes_evt=50265) # all vacob size
        else:
            model = RobertaJointLM.from_pretrained(args.model, mlp_hid=args.mlp_hid_size,
                                                num_classes_tmp=len(label_map_temporal),
                                                num_classes_evt=len(label_map_event))
    else:
        if args.filename1 == "empty":
            model = BertJointLM.from_pretrained(args.model, mlp_hid=args.mlp_hid_size,
                                                   num_classes_tmp=len(label_map_temporal),
                                                   num_classes_evt=30522) # all vocab size
        else:
            model = BertJointLM.from_pretrained(args.model, mlp_hid=args.mlp_hid_size,
                                                num_classes_tmp=len(label_map_temporal),
                                                num_classes_evt=len(label_map_event))

    model.to(device)
        
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    model.train()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps

    if args.fp16:
        try:
            from apex.optimizers import FusedAdam
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    loss_fct = CrossEntropyLoss()
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, tr_loss_tmp, tr_loss_evt = 0.0, 0.0, 0.0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)

            input_ids, input_masks, segment_ids, offsets, tasks, labels = batch

            logits_tmp, logits_evt, embeds = model(input_ids=input_ids, offsets=offsets,
                                                   attention_mask=input_masks, token_type_ids=segment_ids)

            ## Temporal indicator mask loss
            tmp_idx = [k for k, v in enumerate(tasks.tolist()) if v == 1]
            if args.filename1 == "empty":
                assert len(tmp_idx) == 0

            if len(tmp_idx) > 0:
                labels_tmp = torch.stack([labels[i] for i in tmp_idx]).to(device)
                tmp_idx = torch.tensor(tmp_idx).to(device)
                logits_tmp = torch.index_select(logits_tmp, 0, tmp_idx)
                loss_tmp = loss_fct(logits_tmp, labels_tmp)
                tr_loss_tmp += loss_tmp.item()
            else:
                loss_tmp = 0.0

            ## Event mask loss
            evt_idx = [k for k, v in enumerate(tasks.tolist()) if v == 0]
            #print(evt_idx)
            if len(evt_idx) > 0:
                labels_evt = torch.stack([labels[i] for i in evt_idx]).to(device)
                evt_idx = torch.tensor(evt_idx).to(device)
                logits_evt = torch.index_select(logits_evt, 0, evt_idx)
                loss_evt = loss_fct(logits_evt, labels_evt)
                tr_loss_evt += loss_evt.item() * args.event_weight
            else:
                loss_evt = 0.0

            loss = loss_tmp + loss_evt * args.event_weight

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += labels.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                  args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if nb_tr_examples % (args.train_batch_size*1000) == 0:
                logger.info("current train loss is %s" % (tr_loss / float(nb_tr_steps)))
                logger.info("current train temporal loss is %s" % (tr_loss_tmp / float(nb_tr_steps)))
                logger.info("current train event loss is %s" % (tr_loss_evt / float(nb_tr_steps)))

        output_model_file = os.path.join(args.output_dir, "pytorch_model_e%s.bin" % epoch)
        torch.save(model.state_dict(), output_model_file)

if __name__ == "__main__":
    main()
