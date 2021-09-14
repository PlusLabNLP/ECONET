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
import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import *
from models import MultitaskClassifier, MultitaskClassifierRoberta
from utils import *
from optimization import *
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                               Path.home() / '.pytorch_pretrained_roberta'))
PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

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
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters                                                                                            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--instance",
                        type=bool,
                        default=True,
                        help="whether to create sample as instance: 1Q: multiple answers")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--finetune",
                        action='store_true',
                        help="Whether to finetune LM.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model",
                        type=str,
                        help="cosmos_model.bin, te_model.bin",
                        default="")
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
    parser.add_argument("--train_ratio",
                        default=1.0,
                        type=float,
                        help="ratio of training samples")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
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

    # construct model
    if 'roberta' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        if "/%s/" % args.model in args.load_model:
            cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_{}'.format(args.local_rank)
            model = MultitaskClassifierRoberta.from_pretrained(args.model, cache_dir=cache_dir,
                                                               mlp_hid=args.mlp_hid_size)
        else:
            logger.info(args.load_model)
            model_state_dict = torch.load(args.load_model)
            model = MultitaskClassifierRoberta.from_pretrained(args.model, state_dict=model_state_dict,
                                                               mlp_hid=args.mlp_hid_size)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        if "/%s/" % args.model in args.load_model:
            cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank)
            model = MultitaskClassifier.from_pretrained(args.model, cache_dir=cache_dir,
                                                        mlp_hid=args.mlp_hid_size)
        else:
            logger.info(args.load_model)
            model_state_dict = torch.load(args.load_model)
            model = MultitaskClassifier.from_pretrained(args.model, state_dict=model_state_dict,
                                                        mlp_hid=args.mlp_hid_size)
    model.to(device)
    if args.do_train:
        train_data = load_data(args.data_dir, "train", args.file_suffix)
        if 'roberta' in args.model:
            train_features = convert_to_features_roberta(train_data, tokenizer,
                                                         max_length=args.max_seq_length, end_to_end=True)
        else:
            train_features = convert_to_features(train_data, tokenizer,
                                                 max_length=args.max_seq_length, end_to_end=True)

        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'mask_ids'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)

        all_offsets = select_field(train_features, 'offset')
        all_labels  = select_field(train_features, 'label')

        all_key_indices = torch.tensor(list(range(len(all_labels))), dtype=torch.long)
        logger.info("id_size: {} mask_size: {}, instance_key_size: {}, segment_size: {}".format(
            all_input_ids.size(), all_input_mask.size(), all_key_indices.size(), all_segment_ids.size()))

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_key_indices)
        
        # free memory
        del train_features
        del all_input_ids
        del all_input_mask
        del all_segment_ids
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

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

        label_map = {0: 'Negative', 1: 'Positive'}
        
        global_step = 0
        best_eval_f1 = 0.0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, tr_acc = 0.0, 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_masks, segment_ids, instance_indices = batch

                offsets, labels, lengths = flatten_answers([(all_labels[i], all_offsets[i])
                                                            for i in instance_indices.cpu().tolist()])
                labels = torch.tensor(labels).to(device)

                logits, loss = model(input_ids, offsets, lengths, attention_mask=input_masks,
                                     token_type_ids=segment_ids, labels=labels)
                
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

                logits = logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()
                tr_acc += accuracy(logits, labels)

                if nb_tr_examples % 10000 == 0:
                    print("current train loss is %s" % (tr_loss / float(nb_tr_steps)))
                    print("current train accuracy is %s" % (tr_acc / float(nb_tr_examples)))

            if args.do_eval:
                # fix dev data to be "individual_dev_end2end_final"
                eval_data = load_data(args.data_dir, "individual_dev", "_end2end_final.json")
                if 'roberta' in args.model:
                    eval_features = convert_to_features_roberta(eval_data, tokenizer,
                                                                max_length=args.max_seq_length, end_to_end=True,
                                                                evaluation=args.do_eval)
                else:
                    eval_features = convert_to_features(eval_data, tokenizer,
                                                        max_length=args.max_seq_length, end_to_end=True,
                                                        evaluation=args.do_eval)

                eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
                eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                eval_offsets = select_field(eval_features, 'offset')
                eval_labels  = select_field(eval_features, 'label')
                eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

                # collect unique question ids for EM calculation
                question_ids = select_field(eval_features, 'question_id')
                # flatten question_ids
                question_ids = [q for i, q in enumerate(question_ids) for x in range(len(eval_labels[i]))]
                eval_idv_answers = select_field(eval_features, 'individual_answers')

                eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_key_indices)

                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                eval_loss, eval_accuracy, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0, 0
                all_preds, all_golds, max_f1s = [], [], []
                model.eval()

                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.to(device) for t in batch)

                    input_ids, input_masks, segment_ids, instance_indices = batch
                    
                    offsets, labels, lengths = flatten_answers([(eval_labels[i], eval_offsets[i])
                                                            for i in instance_indices.tolist()])
                    all_golds.extend(labels)
                    labels = torch.tensor(labels).to(device)
                    
                    with torch.no_grad():
                        logits, tmp_eval_loss = model(input_ids, offsets, lengths, attention_mask=input_masks,
                                                        token_type_ids=segment_ids, labels=labels)

                    logits = logits.detach().cpu().numpy()
                    labels = labels.to('cpu').numpy()
                    tmp_eval_accuracy = accuracy(logits, labels)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += labels.shape[0]
                    nb_eval_steps += 1

                    batch_preds = np.argmax(logits, axis=1)
                    bi = 0
                    for l, idx in enumerate(instance_indices):
                        pred = [batch_preds[bi + li] for li in range(lengths[l])]
                        pred_names = [label_map[p] for p in pred]
                        max_f1 = 0
                        for gold in eval_idv_answers[idx]:
                            label_names = [label_map[l] for l in gold]
                            if sum(gold) == 0 and sum(pred) == 0:
                                f1 = 1.0
                            else:
                                f1 = cal_f1(pred_names, label_names, {v: k for k, v in label_map.items()})
                            if f1 >= max_f1:
                                max_f1 = f1
                        bi += lengths[l]
                        max_f1s.append(max_f1)                        
                    all_preds.extend(batch_preds)

                assert len(all_preds) == len(question_ids)

                em = exact_match(question_ids, all_golds, all_preds)
                eval_accuracy = eval_accuracy / nb_eval_examples
                label_names = [label_map[l] for l in all_golds]
                pred_names = [label_map[p] for p in all_preds]
                eval_pos_f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})

                logger.info("the current eval accuracy is: %.4f" % eval_accuracy)
                logger.info("the current eval exact match ratio is: %.4f" % em)
                logger.info("the current eval macro positive class F1 is: %.4f" % np.mean(max_f1s))
                logger.info("the current eval micro positive class F1 is: %.4f" % eval_pos_f1)

                if np.mean(max_f1s) > best_eval_f1:
                    logger.info("Save at Epoch %s" % epoch)
                    best_eval_f1 = np.mean(max_f1s)
                    torch.save(model_to_save.state_dict(), output_model_file)

                model.train()

if __name__ == "__main__":
    main()
