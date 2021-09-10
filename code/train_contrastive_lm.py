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
from new_models import ContrastiveLM, ContrastiveLM_Bert
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


# before, after, during, past, next, beginning, ending
temp_groups = [['before', 'previous to', 'prior to', 'preceding', 'followed', 'until'],
               ['after', 'following', 'since', 'soon after', 'once', 'now that'],
               ['during', 'while', 'when', 'the same time', 'at the time', 'meanwhile'],
               ['earlier', 'previously', 'formerly', 'in the past', 'yesterday', 'last time'],
               ['consequently', 'subsequently', 'in turn', 'henceforth', 'later', 'then'],
               ['initially', 'originally', 'at the beginning', 'to begin', 'starting with', 'to start with'],
               ['finally', 'in the end', 'at last', 'lastly']]

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
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--load_model",
                        type=str,
                        help="trained with no contrastive loss",
                        default="")
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
    parser.add_argument("--orig_ratio",
                        default=0.5,
                        type=float,
                        help="original sample ratio")
    parser.add_argument("--group_weight",
                        default=0.5,
                        type=float,
                        help="weight for loss in the same temporal group")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--event_weight",
                        default=1.0,
                        type=float,
                        help="event mask task weight")
    parser.add_argument("--contrastive_weight",
                        default=1.0,
                        type=float,
                        help="contrastive task weight")
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
    label_map_event = {k: v for k, v in enumerate(vocab.keys())}

    logger.info(len(label_map_temporal))
    logger.info(len(label_map_event))

    # temporal masked data
    with open("%s%s" % (args.data_dir, args.filename1)) as infile:
        train_features = json.load(infile)

    # event masked data
    with open("%s%s" % (args.data_dir, args.filename2)) as infile:
        train_features += json.load(infile)

    logger.info("finish loading data...")
    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info(num_train_steps)

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

    if args.load_model:
        logger.info(args.load_model)
        model_state_dict = torch.load(args.load_model)
        if 'roberta' in args.model:
            tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=args.do_lower_case)
            model = ContrastiveLM.from_pretrained(args.model, state_dict=model_state_dict, mlp_hid=args.mlp_hid_size,
                                                  num_classes_tmp=len(label_map_temporal),
                                                  num_classes_evt=len(label_map_event))
        elif 'bert' in args.model:
            tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
            model = ContrastiveLM_Bert.from_pretrained(args.model, state_dict=model_state_dict,
                                                       mlp_hid=args.mlp_hid_size,
                                                       num_classes_tmp=len(label_map_temporal),
                                                       num_classes_evt=len(label_map_event))
        del model_state_dict
        torch.cuda.empty_cache()
    else:
        if 'roberta' in args.model:
            tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
            model = ContrastiveLM.from_pretrained(args.model, mlp_hid=args.mlp_hid_size,
                                                  num_classes_tmp=len(label_map_temporal),
                                                  num_classes_evt=len(label_map_event))
        elif 'bert' in args.model:
            tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
            model = ContrastiveLM_Bert.from_pretrained(args.model, mlp_hid=args.mlp_hid_size,
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
    loss_fct_c = CrossEntropyLoss(reduction='none')

    def is_correct_group(pred, gold):
        for temp_group in temp_groups:
            if label_map_temporal[pred] in temp_group and label_map_temporal[gold] in temp_group:
                return True
        return False

    def create_constrastive_samples(logits, labels, label_map, input_ids, input_masks, segment_ids,
                                    offsets, tokenizer, seed, device, ratio=0.5, group_weight=0.5, max_len=168):
        np.random.seed(seed)
        new_input_ids, new_input_masks, new_segment_ids, new_labels = [], [], [], []
        sample_weights = []
        for b, s in enumerate(np.random.sample(len(logits))):
            # event
            if len(label_map) > 100:
                # use predicted
                if s > ratio:
                    label = np.argmax(logits[b])
                    if label == labels[b]:
                        new_labels.append(1)
                    else:
                        new_labels.append(0)
                # use original
                else:
                    label = labels[b]
                    new_labels.append(1)
                sample_weights.append(1.0)

            # temporal indicator
            else:
                if s > ratio:
                    label = np.argmax(logits[b])
                # adding perturbed samples
                else:
                    label = np.random.randint(0, len(label_map))

                if label == labels[b]:
                    new_labels.append(1)
                    sample_weights.append(1.0)
                else:
                    new_labels.append(0)
                    if is_correct_group(label, labels[b]):
                        sample_weights.append(group_weight)
                    else:
                        sample_weights.append(1.0)

            mask_tok_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label_map[label]))

            # input_ids
            tok_ids = input_ids[b][:offsets[b]] + mask_tok_ids + input_ids[b][offsets[b]+1:]
            padding = [0]*(max_len - len(tok_ids))
            new_input_ids.append(tok_ids + padding)
            # mask_ids
            mask_ids = input_masks[b][:offsets[b]] + [1]*len(mask_tok_ids) + input_masks[b][offsets[b]+1:]
            new_input_masks.append(mask_ids + padding)

            # segment_ids
            seg_ids = segment_ids[b][:offsets[b]] + [0]*len(mask_tok_ids) + segment_ids[b][offsets[b]+1:]
            new_segment_ids.append(seg_ids + padding)

        new_input_ids = torch.tensor(new_input_ids, dtype=torch.long).to(device)
        new_input_masks = torch.tensor(new_input_masks, dtype=torch.long).to(device)
        new_segment_ids = torch.tensor(new_segment_ids, dtype=torch.long).to(device)
        new_labels = torch.tensor(new_labels, dtype=torch.long).to(device)
        sample_weights = torch.tensor(sample_weights, dtype=torch.float).to(device)
        return new_input_ids, new_input_masks, new_segment_ids, new_labels, sample_weights

    def combine_contrastive_samples(input_ids_ctmp, input_masks_ctmp, segment_ids_ctmp, labels_ct, offsets_tmp,
                                    input_ids_cevt, input_masks_cevt, segment_ids_cevt, labels_ce, offsets_evt):

        new_input_ids = torch.cat([input_ids_ctmp, input_ids_cevt], dim=0)
        new_input_masks = torch.cat([input_masks_ctmp, input_masks_cevt], dim=0)
        new_segment_ids = torch.cat([segment_ids_ctmp, segment_ids_cevt], dim=0)
        new_labels = torch.cat([labels_ct, labels_ce], dim=0)
        new_offsets = torch.cat([offsets_tmp, offsets_evt], dim=0)

        return new_input_ids, new_input_masks, new_segment_ids, new_labels, new_offsets


    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, tr_loss_tmp, tr_loss_evt, tr_loss_contrasitve_t, tr_loss_contrasitve_e = 0.0, 0.0, 0.0, 0.0, 0.0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            input_ids, input_masks, segment_ids, offsets, tasks, labels = batch
            logits_tmp, logits_evt, embeds = model(input_ids.to(device), offsets.to(device),
                                                   attention_mask=input_masks.to(device),
                                                   token_type_ids=segment_ids.to(device), task='mask')

            ### Temporal Indicator
            tmp_idx = [k for k, v in enumerate(tasks.tolist()) if v == 1]
            if len(tmp_idx) > 0:
                ## Temporal indicator mask loss
                labels_tmp = torch.stack([labels[i] for i in tmp_idx])
                offsets_tmp = torch.stack([offsets[i] for i in tmp_idx])
                tmp_idx = torch.tensor(tmp_idx).to(device)
                logits_tmp = torch.index_select(logits_tmp, 0, tmp_idx)

                # Contrastive input
                input_ids_ctmp, input_masks_ctmp, segment_ids_ctmp, labels_ct, sample_weights_t = \
                    create_constrastive_samples(logits_tmp.tolist(), labels_tmp.tolist(), label_map_temporal,
                                                input_ids.tolist(), input_masks.tolist(), segment_ids.tolist(),
                                                offsets_tmp.tolist(), tokenizer, args.seed, device,
                                                args.orig_ratio, args.group_weight)

                # Loss computation
                loss_tmp = loss_fct(logits_tmp, labels_tmp.to(device))
                tr_loss_tmp += loss_tmp.item()
            else:
                loss_tmp = 0.0

            ### Event
            evt_idx = [k for k, v in enumerate(tasks.tolist()) if v == 0]
            if len(evt_idx) > 0:
                ## Event mask loss
                labels_evt = torch.stack([labels[i] for i in evt_idx])
                offsets_evt = torch.stack([offsets[i] for i in evt_idx])
                evt_idx = torch.tensor(evt_idx).to(device)
                logits_evt = torch.index_select(logits_evt, 0, evt_idx)

                # Contrastive Loss
                input_ids_cevt, input_masks_cevt, segment_ids_cevt, labels_ce, sample_weights_e = \
                    create_constrastive_samples(logits_evt.tolist(), labels_evt.tolist(), label_map_event,
                                                input_ids.tolist(), input_masks.tolist(), segment_ids.tolist(),
                                                offsets_evt.tolist(), tokenizer, args.seed, device, args.orig_ratio)
                # Loss computation
                loss_evt = loss_fct(logits_evt, labels_evt.to(device))
                tr_loss_evt += loss_evt.item() * args.event_weight
            else:
                loss_evt = 0.0

            ### Now combine constrastive samples and compute combined loss
            if len(tmp_idx) == 0:
                input_ids_c = input_ids_cevt
                input_masks_c = input_masks_cevt
                segment_ids_c = segment_ids_cevt
                labels_c = labels_ce
                _, logits_c = model(input_ids_c, offsets_evt.to(device), attention_mask=input_masks_c,
                                    token_type_ids=segment_ids_c, task='contrastive')
                loss_c = torch.mean(loss_fct_c(logits_c, labels_c) * sample_weights_e)
                tr_loss_contrasitve_e += loss_c.item() * args.contrastive_weight
            elif len(evt_idx) == 0:
                input_ids_c = input_ids_ctmp
                input_masks_c = input_masks_ctmp
                segment_ids_c = segment_ids_ctmp
                labels_c = labels_ct
                logits_c, _ = model(input_ids_c, offsets_tmp.to(device), attention_mask=input_masks_c,
                                    token_type_ids=segment_ids_c, task='contrastive')
                loss_c = torch.mean(loss_fct_c(logits_c, labels_c) * sample_weights_t)
                tr_loss_contrasitve_t += loss_c.item() * args.contrastive_weight
            else:
                input_ids_c, input_masks_c, segment_ids_c, labels_c, offsets_c = \
                    combine_contrastive_samples(input_ids_ctmp, input_masks_ctmp, segment_ids_ctmp, labels_ct, offsets_tmp,
                                                input_ids_cevt, input_masks_cevt, segment_ids_cevt, labels_ce, offsets_evt)
                logits_ct, logits_ce = model(input_ids_c, offsets_c.to(device), attention_mask=input_masks_c,
                                             token_type_ids=segment_ids_c, task='contrastive')
                loss_ct = torch.mean(loss_fct_c(logits_ct[:len(tmp_idx)], labels_ct) * sample_weights_t)
                loss_ce = torch.mean(loss_fct_c(logits_ce[len(tmp_idx):], labels_ce) * sample_weights_e)
                loss_c = loss_ct + loss_ce
                tr_loss_contrasitve_t += loss_ct.item() * args.contrastive_weight
                tr_loss_contrasitve_e += loss_ce.item() * args.contrastive_weight

            loss = loss_tmp + loss_evt * args.event_weight + args.contrastive_weight * loss_c
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

            if nb_tr_examples % (args.train_batch_size*2000) == 0:
                logger.info("current train loss is %s" % (tr_loss / float(nb_tr_steps)))
                logger.info("current temporal loss is %s" % (tr_loss_tmp / float(nb_tr_steps)))
                logger.info("current event loss is %s" % (tr_loss_evt / float(nb_tr_steps)))
                logger.info("current temporal contrastive loss is %s" % (tr_loss_contrasitve_t / float(nb_tr_steps)))
                logger.info("current event contrastive loss is %s" % (tr_loss_contrasitve_e / float(nb_tr_steps)))

        output_model_file = os.path.join(args.output_dir, "pytorch_model_e%s.bin" % epoch)
        torch.save(model.state_dict(), output_model_file)

if __name__ == "__main__":
    main()
