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
from models import TEClassifierRoberta, TEClassifier
from optimization import *
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
    parser.add_argument("--te_type",
                        default=None,
                        type=str,
                        required=True,
                        help="subfolder contains TE data")
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

    if args.te_type in ['tbd']:
        label_map = tbd_label_map
    if args.te_type in ['matres']:
        label_map = matres_label_map
    if args.te_type in ['red']:
        label_map = red_label_map
    num_classes = len(label_map)

    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin")
    if 'roberta' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_-1'
        model = TEClassifierRoberta.from_pretrained(args.model, state_dict=model_state_dict,
                                                    cache_dir=cache_dir, mlp_hid=args.mlp_hid_size,
                                                    num_classes=num_classes)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_-1'
        model = TEClassifier.from_pretrained(args.model, state_dict=model_state_dict,
                                             cache_dir=cache_dir, mlp_hid=args.mlp_hid_size,
                                              num_classes=num_classes)

    model.to(device)
    for eval_file in ['dev', 'test']:
        if args.te_type in ["matres"]:
            if eval_file == 'dev':
                trainIds, devIds = get_train_dev_ids(args.data_dir, args.te_type)
                eval_features_te = convert_examples_to_features_te(args.data_dir, args.te_type, 'train',
                                                                    tokenizer, args.max_seq_length, True, devIds)
            else:
                eval_features_te = convert_examples_to_features_te(args.data_dir, args.te_type, eval_file,
                                                                    tokenizer, args.max_seq_length, True)
        else:
            eval_features_te = convert_examples_to_features_te(args.data_dir, args.te_type, eval_file,
                                                                tokenizer, args.max_seq_length, True,
                                                                analyze=args.analyze)
        te_sample_size = len(eval_features_te)
        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Num TE examples = %d", len(eval_features_te))
        
        eval_input_ids_te = torch.tensor(select_field_te(eval_features_te, 'input_ids'), dtype=torch.long)
        eval_input_mask_te = torch.tensor(select_field_te(eval_features_te, 'input_mask'), dtype=torch.long)
        eval_segment_ids_te = torch.tensor(select_field_te(eval_features_te, 'segment_ids'), dtype=torch.long)
        eval_lidx_s = torch.tensor(select_field_te(eval_features_te, 'lidx_s'), dtype=torch.long)
        eval_lidx_e = torch.tensor(select_field_te(eval_features_te, 'lidx_e'), dtype=torch.long)
        eval_ridx_s = torch.tensor(select_field_te(eval_features_te, 'ridx_s'), dtype=torch.long)
        eval_ridx_e = torch.tensor(select_field_te(eval_features_te, 'ridx_e'), dtype=torch.long)
        eval_pred_inds = torch.tensor(select_field_te(eval_features_te, 'pred_ind'), dtype=torch.long)
        eval_label_te = torch.tensor([f.label for f in eval_features_te], dtype=torch.long)
        eval_input_length_te = torch.tensor([f.length for f in eval_features_te], dtype=torch.long)

        eval_data = TensorDataset(eval_input_ids_te, eval_input_mask_te, eval_segment_ids_te, eval_label_te,
                                  eval_lidx_s, eval_lidx_e, eval_ridx_s, eval_ridx_e, eval_pred_inds,
                                  eval_input_length_te)
        
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.analyze:
            temp_group_inds = [f.temp_group_ind for f in eval_features_te]

        all_preds, te_preds, all_golds = [], [], []

        idx2label = {k: v for k, v in enumerate(label_map)}
        te_true_labels = [idx2label[f.label] for f in eval_features_te[:te_sample_size]]
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids_te, input_mask_te, segment_ids_te, labels_te, \
                lidx_s, lidx_e, ridx_s, ridx_e, pred_ind, length_te = batch
            with torch.no_grad():
                loss_te, logit_te = model(input_ids_te, token_type_ids_te=segment_ids_te,
                                          attention_mask_te=input_mask_te, lidx_s=lidx_s, lidx_e=lidx_e,
                                          ridx_s=ridx_s, ridx_e=ridx_e, length_te=length_te,
                                          labels_te=labels_te)

                logit_te = logit_te.detach().cpu().numpy()
                pred_te = np.argmax(logit_te, axis=1).tolist()
                te_preds.extend(pred_te)

        te_preds_labels = [idx2label[x] for x in te_preds[:te_sample_size]]
        report = ClassificationReport(args.task_name + "-" + args.te_type, te_true_labels, te_preds_labels)
        print(report)

if __name__ == "__main__":
    main()
