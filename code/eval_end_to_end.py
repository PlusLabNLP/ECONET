
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
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
from models import MultitaskClassifier, MultitaskClassifierRoberta
from optimization import *
from collections import defaultdict
from utils import *
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
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where the trained model are saved")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="Suffix of filename")
    ## Other parameters                                                                                            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--eval_ratio",
                        default=0.5,
                        type=float,
                        help="portion of data for evaluation")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=7,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    logger.info("current task is " + str(task_name))

    label_map = {0: 'Negative', 1: 'Positive'}
    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin")

    if 'roberta' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_-1'
        model = MultitaskClassifierRoberta.from_pretrained(args.model, state_dict=model_state_dict,
                                                           cache_dir=cache_dir, mlp_hid=args.mlp_hid_size)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_-1'
        model = MultitaskClassifier.from_pretrained(args.model, state_dict=model_state_dict,
                                                    cache_dir=cache_dir, mlp_hid=args.mlp_hid_size)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
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
    
    for eval_file in ['dev']:
        model.eval()
        print("=" * 50 + "Evaluating %s" % eval_file + "="* 50)
        eval_data = load_data(args.data_dir, "individual_%s" % eval_file, args.file_suffix)
        if 'roberta' in args.model:
            eval_features = convert_to_features_roberta(eval_data, tokenizer, max_length=args.max_seq_length,
                                                        evaluation=True, end_to_end=True)
        else:
            eval_features = convert_to_features(eval_data, tokenizer, max_length=args.max_seq_length,
                                                        evaluation=True, end_to_end=True)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
        eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        eval_offsets = select_field(eval_features, 'offset')
        eval_labels  = select_field(eval_features, 'label')
        eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)
        
        # collect unique question ids for EM calculation
        question_ids = select_field(eval_features, 'question_id')
        question_ids = [q for i, q in enumerate(question_ids) for x in range(len(eval_labels[i]))]
        # collect unique question culster for EM-cluster calculation                          
        question_cluster = select_field(eval_features, 'question_cluster')
        question_cluster_size = select_field(eval_features, 'cluster_size')
        eval_idv_answers = select_field(eval_features, 'individual_answers')

        eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_key_indices)
                
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss, eval_accuracy, best_eval_f1, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0.0, 0, 0
        all_preds, all_golds, max_f1s, macro_f1s = [], [], [], []
        f1_dist = defaultdict(list)
        em_counter = 0
        em_cluster_agg, em_cluster_relaxed, f1_cluster_80 = {}, {}, {}

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

            nb_eval_examples += labels.shape[0]
            nb_eval_steps += 1

            batch_preds = np.argmax(logits, axis=1)
            bi = 0
            for l, idx in enumerate(instance_indices):
                pred = [batch_preds[bi + li] for li in range(lengths[l])]
                pred_names = [label_map[p] for p in pred]
                gold_names = [label_map[labels[bi + li]] for li in range(lengths[l])]
                is_em = (pred_names == gold_names)

                if sum([labels[bi + li] for li in range(lengths[l])]) == 0 and sum(pred) == 0:
                    macro_f1s.append(1.0)
                else:
                    macro_f1s.append(cal_f1(pred_names, gold_names, {v:k for k,v in label_map.items()}))

                # Each instance is annotated multiple times, pick the highest score from model (Relaxed)
                max_f1, instance_matched = 0, 0
                for gold in eval_idv_answers[idx]:
                    label_names = [label_map[l] for l in gold]
                    if pred_names == label_names: instance_matched = 1
                    if sum(gold) == 0 and sum(pred) == 0:
                        f1 = 1.0
                    else:
                        f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})
                    if f1 >= max_f1:
                        max_f1 = f1
                        key = len(gold)

                if question_cluster_size[idx] > 1:
                    if question_cluster[idx] not in em_cluster_agg:
                        em_cluster_agg[question_cluster[idx]] = 1
                    if is_em == 0: em_cluster_agg[question_cluster[idx]] = 0
                        
                    if question_cluster[idx] not in em_cluster_relaxed:
                        em_cluster_relaxed[question_cluster[idx]] = 1
                    if instance_matched == 0: em_cluster_relaxed[question_cluster[idx]] = 0

                    if question_cluster[idx] not in f1_cluster_80:
                        f1_cluster_80[question_cluster[idx]] = 1
                    if max_f1 < 0.8: f1_cluster_80[question_cluster[idx]] = 0
                        
                bi += lengths[l]
                max_f1s.append(max_f1)
                em_counter += instance_matched
                f1_dist[key].append(max_f1)

            all_preds.extend(batch_preds)

        assert len(em_cluster_relaxed) == len(em_cluster_agg)
        assert len(f1_cluster_80) == len(em_cluster_agg) 
        
        em_cluster_relaxed_res = sum(em_cluster_relaxed.values()) / len(em_cluster_relaxed)
        em_cluster_agg_res = sum(em_cluster_agg.values()) / len(em_cluster_agg)
        f1_cluster_80_res = sum(f1_cluster_80.values()) / len(f1_cluster_80)
        
        label_names = [label_map[l] for l in all_golds]
        pred_names = [label_map[p] for p in all_preds]
        
        # question_id is also flattened
        em = exact_match(question_ids, label_names, pred_names)
        eval_pos_f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})

        print("the current eval positive class Micro F1 (Agg) is: %.4f" % eval_pos_f1)
        print("the current eval positive class Macro F1 (Relaxed) is: %.4f" % np.mean(max_f1s))
        print("the current eval positive class Macro F1 (Agg) is: %.4f" % np.mean(macro_f1s))

        print("the current eval exact match (Agg) ratio is: %.4f" % em)
        print("the current eval exact match ratio (Relaxed) is: %.4f" % (em_counter / len(eval_features)))

        print("%d Clusters" % len(em_cluster_relaxed))
        print("the current eval clustered EM (Agg) is: %.4f" % (em_cluster_agg_res))
        print("the current eval clustered EM (Relaxed) is: %.4f" % (em_cluster_relaxed_res))
        print("the current eval clusrered F1 (max>=0.8) is: %.4f" % (f1_cluster_80_res))

if __name__ == "__main__":
    main()
