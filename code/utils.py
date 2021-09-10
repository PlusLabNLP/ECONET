from typing import Iterator, List, Mapping, Union, Optional, Set
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
import json
import pickle
import numpy as np
import logging
import lxml

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir, split, suffix):
    filename = "%s%s%s" % (data_dir, split, suffix)
    print("==========load data from %s ===========" % filename)
    with open(filename, "r") as read_file:
        return json.load(read_file)

def select_field(data, field):
    # collect a list of field in data                                                                  
    # fields: 'label', 'offset', 'input_ids, 'mask_ids', 'segment_ids', 'question_id'                            
    return [ex[field] for ex in data]

def select_field_te(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def exact_match(question_ids, labels, predictions):
    em = defaultdict(list)
    for q, l, p in zip(question_ids, labels, predictions):
        em[q].append(l == p)
    print("Total %s questions" % len(em))
    return float(sum([all(v) for v in em.values()])) / float(len(em))

def sample_errors(passages, questions, answers, labels, preds, label_class="Positive", num=50):
    assert len(passages) == len(preds)
    assert len(questions) == len(preds)
    assert len(answers) == len(preds)
    errors = []
    outfile = open("%s_error_samples.tsv" % label_class, 'w')
    outfile.write("Passage\tQuestion\tAnswer-span\tAnswer-offset\tAnswer-label\tAnswer-prediction\n")
    count = 0
    for pa, q, a, l, p in zip(passages, questions, answers, labels, preds):
        if count >= num:
            continue
        if l == label_class and l != p:
            outfile.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (" ".join(pa), q, a['span'], a['idx'], l, p))
            count += 1
    outfile.close()
    return

def get_train_dev_ids(data_dir, data_type):
    trainIds = [f.strip() for f in open("%s/%s/trainIds.txt" % (data_dir, data_type))]
    devIds = [f.strip() for f in open("%s/%s/devIds.txt" % (data_dir, data_type))]
    return trainIds, devIds

def convert_to_features_roberta(data, tokenizer, max_length=150, evaluation=False,
                                end_to_end=False):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 

    for k, v in data.items():
        segment_ids = []
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])

        new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            
        segment_ids = [0] * len(tokenized_ids)
        # mask ids                                                                                            
        mask_ids = [1] * len(tokenized_ids)

         # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}

            # add these three field for qualitative analysis                                            
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                sample['answer'] = v['answers']
                sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            samples.append(sample)
        else:
            # no duplicate P + Q
            labels, offsets = [], []
            for vv in v['answers'].values():
                labels.append(vv['label'])
                offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}
            
            # add these three field for qualitative analysis         
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                individual_answers = []
                for vv in v['individual_answers']:
                    individual_answers.append([a['label'] for a in vv.values()])
                sample['individual_answers'] = individual_answers
                
            samples.append(sample)

        # check some example data
        if counter < 0:
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples


def convert_to_features_event(data, tokenizer, max_length=150):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0  # to show global max_len without truncating

    for k, v in data.items():
        new_tokens = [tokenizer.cls_token]  # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
        new_tokens.append(tokenizer.sep_token)
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1  # account for ending </s>

        tokenized_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]

        segment_ids = [0] * len(tokenized_ids)
        # mask ids
        mask_ids = [1] * len(tokenized_ids)

        # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> counted in orig_to_tok_map already, so only need to worry about <s>
        labels, offsets = [], []
        for kk, vv in enumerate(v['answers']['types']):
            labels.append(vv)
            offsets.append(orig_to_tok_map[kk])

        sample = {'label': labels,
                  'offset': offsets,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k}
        samples.append(sample)

        # check some example data
        if counter < 0:
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples


def flatten_answers(answers):    
    # flatten answers and use batch length to map back to the original input   
    offsets = [a for ans in answers for a in ans[1]]
    labels = [a for ans in answers for a in ans[0]]
    lengths = [len(ans[0]) for ans in answers]

    assert len(offsets)  == sum(lengths)
    assert len(labels) == sum(lengths)
    
    return offsets, labels, lengths


def convert_to_features_longformer(data, global_idx, documents, tokenizer, global_context_win=1,
                                   max_length=256, evaluation=False, end_to_end=False):
    # Input: <s> Question </s> Global_Context_prev </s> Local_Context </s> Global_Context_post </s>
    # segID: 00000000000000000 1111111111111111111 22222222222222222222222 11111111111111111111111
    samples = []
    counter = 0
    max_len_global = 0  # to show global max_len without truncating
    end = -3 if evaluation else -4

    for k, v in data.items():
        doc_id = "_".join(k.split("_")[1:end])
        global_sent_idx = global_idx[k]
        document = documents[doc_id]

        local_start, local_end = global_sent_idx['sent_start'], global_sent_idx['sent_end']

        global_mask_ids = []
        # 1. Question
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])
        global_mask_ids += [2]*(len(question) + 2) # account for <s> </s>

        # 2. Previous Global Context
        new_tokens = ["</s>", "</s>"]  # two sent sep symbols according to the huggingface documentation
        prev_global_context = [t for sent in
                               document[local_start - global_context_win:local_start]
                               for x in sent for t in tokenizer.tokenize(x)]
        new_tokens += prev_global_context
        global_mask_ids += [0] * (len(prev_global_context) + 2)

        orig_to_tok_map = []
        # 3. Local Context
        new_tokens += ["</s>", "</s>"]
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            global_mask_ids += [2] * len(temp_tokens)
            new_tokens += temp_tokens

        new_tokens += ["</s>"]
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        global_mask_ids += [2] * 2 # account for </s> </s>
        assert len(orig_to_tok_map) == len(v['context']) + 1  # account for ending </s>

        # 4. Post Global Context
        new_tokens += ["</s>"]  # two sent sep symbols according to the huggingface documentation
        post_global_context = [t for sent in
                               document[local_end + 1:local_end + global_context_win + 1]
                               for x in sent for t in tokenizer.tokenize(x)]
        new_tokens += post_global_context
        global_mask_ids += [0] * (len(post_global_context) + 2)
        new_tokens += ["</s>"]

        # Combine
        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        assert len(tokenized_ids) == len(global_mask_ids)

        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            global_mask_ids = global_mask_ids[:max_length]

        segment_ids = [0] * len(tokenized_ids)
        # mask ids
        mask_ids = [1] * len(tokenized_ids)

        # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
            global_mask_ids += padding

        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'global_mask_ids': global_mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}
            # add these three field for qualitative analysis
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                sample['answer'] = v['answers']
                sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            samples.append(sample)
        else:
            # no duplicate P + Q
            labels, offsets = [], []
            for vv in v['answers'].values():
                labels.append(vv['label'])
                offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}

            # add these three field for qualitative analysis
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                individual_answers = []
                for vv in v['individual_answers']:
                    individual_answers.append([a['label'] for a in vv.values()])
                sample['individual_answers'] = individual_answers

            samples.append(sample)

        # check some example data
        if counter < 0:
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples

def convert_to_features(data, tokenizer, max_length=150, evaluation=False,
                        instance=True, end_to_end=False):
    # each sample will have [CLS] + Question + [SEP] + Context                                                                                 
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating                                                                             
    for k, v in data.items():
        segment_ids = []
        start_token = ['[CLS]']
        question = tokenizer.tokenize(v['question'])

        # the following bert tokenized context starts / end with ['SEP']                                                                       
        new_tokens = ["[SEP]"]
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("[SEP]")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending ['SEP']                                                      

        # following the bert convention for calculating segment ids                                                                            
        segment_ids = [0] * (len(question) + 2) + [1] * (len(new_tokens) - 1)

        # mask ids                                                                                                                             
        mask_ids = [1] * len(segment_ids)

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        assert len(tokenized_ids) == len(segment_ids)

        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        # truncate long sequence, but we can simply set max_length > global_max                                                                
        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            mask_ids = mask_ids[:max_length]
            segment_ids = segment_ids[:max_length]
            assert len(tokenized_ids) == max_length

        # padding                                                                                                                              
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.                                                                                              
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding

        if end_to_end:
            labels, offsets = [], []
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}
            # add these three field for qualitative analysis                                                                
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                sample['answer'] = v['answers']
                sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            samples.append(sample)
        else:
            # no duplicate P + Q                                                  
            labels, offsets = [], []
            for vv in v['answers'].values():
                labels.append(vv['label'])
                offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}

            # add these three field for qualitative analysis            
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
                individual_answers = []
                for vv in v['individual_answers']:
                    individual_answers.append([a['label'] for a in vv.values()])
                sample['individual_answers'] = individual_answers
            samples.append(sample)
            
        # check some example data                                                                                                              
        if counter < 0:
            print(k)
            print(v)
            print(tokenized_ids)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples

def cal_f1(pred_labels, true_labels, label_map, log=False):
    def safe_division(numr, denr, on_err=0.0):
        return on_err if denr == 0.0 else numr / denr

    assert len(pred_labels) == len(true_labels)

    total_true = Counter(true_labels)
    total_pred = Counter(pred_labels)

    labels = list(label_map)

    n_correct = 0
    n_true = 0
    n_pred = 0

    # we only need positive f1 score
    exclude_labels = ['Negative']
    for label in labels:
        if label not in exclude_labels:
            true_count = total_true.get(label, 0)
            pred_count = total_pred.get(label, 0)

            n_true += true_count
            n_pred += pred_count

            correct_count = len([l for l in range(len(pred_labels))
                                 if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
            n_correct += correct_count

    precision = safe_division(n_correct, n_pred)
    recall = safe_division(n_correct, n_true)
    f1_score = safe_division(2.0 * precision * recall, precision + recall)
    if log:
        logger.info("Correct: %d\tTrue: %d\tPred: %d" % (n_correct, n_true, n_pred))
        logger.info("Overall Precision: %.4f\tRecall: %.4f\tF1: %.4f" % (precision, recall, f1_score))
    return f1_score


tbd_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS'),
                             ('INCLUDES', 'INCLUDES'),
                             ('IS_INCLUDED', 'IS_INCLUDED'),
                             ])

matres_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                                ('BEFORE', 'BEFORE'),
                                ('AFTER', 'AFTER'),
                                ('SIMULTANEOUS', 'SIMULTANEOUS')
                                ])

red_label_map = OrderedDict([
    #('TERMINATES', 'TERMINATES'),
    ('SIMULTANEOUS', 'OVERLAP'),
    #('REINITIATES',  'REINITIATES'),
    ('OVERLAP/PRECONDITION', 'OVERLAP'),
    ('OVERLAP/CAUSES', 'OVERLAP'),
    ('OVERLAP', 'OVERLAP'),
    #('INITIATES', 'INITIATES'),
    ('ENDS-ON', 'BEFORE'),
    #('CONTINUES', 'CONTINUES'),
    ('CONTAINS-SUBEVENT', 'INCLUDES'),
    ('CONTAINS', 'INCLUDES'),
    #('IS_INCLUDED', 'IS_INCLUDED'),
    ('BEGINS-ON', 'OVERLAP'),
    ('BEFORE/PRECONDITION', 'BEFORE'),
    ('BEFORE/CAUSES', 'BEFORE'),
    ('BEFORE', 'BEFORE'),
    #('AFTER', 'AFTER')
])

class Event():
    def __init__(self, id, type, text, tense, polarity, span):
        self.id = id
        self.type = type
        self.text = text
        self.tense = tense
        self.polarity = polarity
        self.span = span

class TEFeatures(object):
    def __init__(self,
                 example_id,
                 length,
                 doc_id,
                 features,
                 label,
                 temp_group_ind
                 ):
        self.example_id = example_id
        self.length = length
        self.doc_id = doc_id
        self.choices_features = [
            {
                'input_ids': features[0],
                'input_mask': features[1],
                'segment_ids': features[2],
                'left_id': features[3],
                'right_id': features[4],
                'lidx_s': features[5],
                'lidx_e': features[6],
                'ridx_s': features[7],
                'ridx_e': features[8],
                'pred_ind': features[9],
                'sample_counter': features[10]
            }
        ]
        self.label = label
        self.temp_group_ind = temp_group_ind


temp_groups = [['before', 'previous to', 'prior to', 'preceding', 'followed', 'until'],
               ['after', 'following', 'since', 'soon after', 'once', 'now that'],
               ['during', 'while', 'when', 'the same time', 'at the time', 'meanwhile'],
               ['earlier', 'previously', 'formerly', 'in the past', 'yesterday', 'last time'],
               ['consequently', 'subsequently', 'in turn', 'henceforth', 'later', 'then'],
               ['initially', 'originally', 'at the beginning', 'to begin', 'starting with', 'to start with'],
               ['finally', 'in the end', 'at last', 'lastly']]

temp_groups_names = ['before', 'after', 'during', 'past', 'future', 'begin', 'end']

def convert_examples_to_features_te(data_dir, data_type, split, tokenizer, max_seq_length,
                                            is_training, includeIds=None, analyze=False):
    """Loads a data file into a list of InputBatch"""
    if data_type == "matres":
        label_map = matres_label_map
    elif data_type == "tbd":
        label_map = tbd_label_map
    elif data_type == "red":
        label_map = red_label_map

    all_labels = list(OrderedDict.fromkeys(label_map.keys()))
    label_to_id = OrderedDict([(all_labels[l], l) for l in range(len(all_labels))])
    id_to_label = OrderedDict([(l, all_labels[l]) for l in range(len(all_labels))])

    print(label_to_id)
    print(id_to_label)
    examples = pickle.load(open("%s/%s/%s.pickle" % (data_dir, data_type, split), "rb"))
    count, counter, global_max = 0, 0, 0
    features, lengths = [], []
    label_counter = Counter()
    category_counter = Counter()
    for ex_id, ex in examples.items():
        # bidirectional loss + transformer seems to tackle the reverse pair issue
        if '_rev' in ex['rel_type']:
            label = ex['rel_type'][:-4]
        else:
            label = ex['rel_type']
        if label not in label_map:
            # print(label)
            continue

        label_counter[label] += 1

        label_id = label_to_id[label]
        doc_id = ex['doc_id']

        # handle train / dev for matres
        if includeIds and doc_id not in includeIds:
            continue

        left_id = ex['left_event'].id
        right_id = ex['right_event'].id

        pos_dict = ex['doc_dictionary']

        group_ind = [0]*8
        if analyze:
            n_match = 0
            for v in pos_dict.values():
                for i, group in enumerate(temp_groups):
                    if v[0].lower() in group:
                        ti = i
                        n_match += 1
                        group_ind[ti] = 1
                        category_counter[temp_groups_names[ti]] += 1

            if n_match == 0:
                group_ind[-1] = 1
                category_counter['None'] += 1

        all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex['left_event'].span,
                                                                         ex['right_event'].span,
                                                                         pos_dict)
        left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
        right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]
        in_seq = [pos_dict[x][0] for x in all_keys[lidx_start:ridx_end + 1]]

        try:
            sent_start = max(loc for loc, val in enumerate(left_seq) if val == '.') + 1
        except:
            sent_start = 0

        try:
            sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == '.')
        except:
            sent_end = len(pos_dict)

        assert sent_start < sent_end
        assert sent_start <= lidx_start
        assert ridx_end <= sent_end

        # if > 2 sentences, not predicting
        pred_ind = True

        sent_key = all_keys[sent_start:sent_end]
        orig_sent = [pos_dict[x][0].lower() for x in sent_key]

        lidx_start_s = lidx_start - sent_start
        lidx_end_s = lidx_end - sent_start
        ridx_start_s = ridx_start - sent_start
        ridx_end_s = ridx_end - sent_start

        mask_ids = []

        new_tokens = [tokenizer.cls_token]
        orig_to_tok_map = []

        mask_ids.append(1)
        for i, token in enumerate(orig_sent):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            for t in temp_tokens:
                mask_ids.append(1)

        length = len(new_tokens)
        orig_to_tok_map.append(length)

        length += 1
        if length > global_max:
            global_max = length

        # truncate if token lenght exceed max and set pred_ind = False
        if len(new_tokens) + 1 > max_seq_length:
            new_tokens = new_tokens[:max_seq_length - 1]
            mask_ids = mask_ids[:max_seq_length - 1]
            pred_ind = False
            length = max_seq_length
        lengths.append(length)

        # append ending
        new_tokens.append(tokenizer.sep_token)
        mask_ids.append(1)
        assert len(mask_ids) == len(new_tokens)

        # padding
        new_tokens += [tokenizer.pad_token] * (max_seq_length - len(new_tokens))
        mask_ids += [0] * (max_seq_length - len(mask_ids))

        # map original token index into bert (word_piece) index
        lidx_start_s = orig_to_tok_map[lidx_start_s]
        lidx_end_s = orig_to_tok_map[lidx_end_s + 1] - 1

        ridx_start_s = orig_to_tok_map[ridx_start_s]
        ridx_end_s = orig_to_tok_map[ridx_end_s + 1] - 1

        ## a quick trick to tackle long sents: use last token
        if pred_ind == False:
            if ridx_end_s >= max_seq_length:
                ridx_start_s = max_seq_length - 1
                ridx_end_s = max_seq_length - 1
                count += 1
            if lidx_end_s >= max_seq_length:
                lidx_start_s = max_seq_length - 1
                lidx_end_s = max_seq_length - 1

        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        # only one segment for Roberta
        segments_ids = [0] * len(input_ids)

        assert len(input_ids) == len(segments_ids)
        assert len(input_ids) == len(mask_ids)

        features.append(TEFeatures(ex_id, length, doc_id,
                                   (input_ids, mask_ids, segments_ids, left_id, right_id,
                                    lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind, counter),
                                    label_id, group_ind)
                        )
        counter += 1
    print(label_counter)
    print(category_counter)

    print("%d sentences have more than %d tokens, %d" % (
        sum([v for k, v in Counter(lengths).items() if k >= max_seq_length]), max_seq_length, count))
    print("TE global max: %s" % global_max)
    return features

def token_idx(left, right, pos_dict):
    all_keys = list(pos_dict.keys())

    ### to handle case with multiple tokens
    lkey_start = str(left[0])
    lkey_end = str(left[1])

    ### to handle start is not an exact match -- "tomtake", which should be "to take"
    lidx_start = 0
    while int(all_keys[lidx_start].split(':')[1][:-1]) <= left[0]:
        lidx_start += 1

    ### to handle case such as "ACCOUNCED--" or multiple token ends with not match
    lidx_end = lidx_start
    try:
        while left[1] > int(all_keys[lidx_end].split(':')[1][:-1]):
            lidx_end += 1
    except:
        lidx_end -= 1

    ridx_start = 0
    while int(all_keys[ridx_start].split(':')[1][:-1]) <= right[0]:
        ridx_start += 1

    ridx_end = ridx_start
    try:
        while right[1] > int(all_keys[ridx_end].split(':')[1][:-1]):
            ridx_end += 1
    except:
        ridx_end -= 1
    return all_keys, lidx_start, lidx_end, ridx_start, ridx_end

class ClassificationReport:
    def __init__(self, name, true_labels: List[Union[int, str]],
                 pred_labels: List[Union[int, str]]):

        assert len(true_labels) == len(pred_labels)
        self.num_tests = len(true_labels)
        self.total_truths = Counter(true_labels)
        self.total_predictions = Counter(pred_labels)
        self.name = name
        self.labels = sorted(set(true_labels) | set(pred_labels))
        self.confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        self.accuracy = sum(y == y_ for y, y_ in zip(true_labels, pred_labels)) / len(true_labels)
        self.trim_label_width = 15
        self.rel_f1 = 0.0
        self.res_dict = {}

    @staticmethod
    def confusion_matrix(true_labels: List[str], predicted_labels: List[str]) \
            -> Mapping[str, Mapping[str, int]]:
        mat = defaultdict(lambda: defaultdict(int))
        for truth, prediction in zip(true_labels, predicted_labels):
            mat[truth][prediction] += 1
        return mat

    def __repr__(self):
        res = f'Name: {self.name}\t Created: {datetime.now().isoformat()}\t'
        res += f'Total Labels: {len(self.labels)} \t Total Tests: {self.num_tests}\n'
        display_labels = [label[:self.trim_label_width] for label in self.labels]
        label_widths = [len(l) + 1 for l in display_labels]
        max_label_width = max(label_widths)
        header = [l.ljust(w) for w, l in zip(label_widths, display_labels)]
        header.insert(0, ''.ljust(max_label_width))
        res += ''.join(header) + '\n'
        for true_label, true_disp_label in zip(self.labels, display_labels):
            predictions = self.confusion_mat[true_label]
            row = [true_disp_label.ljust(max_label_width)]
            for pred_label, width in zip(self.labels, label_widths):
                row.append(str(predictions[pred_label]).ljust(width))
            res += ''.join(row) + '\n'
        res += '\n'

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        def num_to_str(num):
            return '0' if num == 0 else str(num) if type(num) is int else f'{num:.4f}'

        n_correct = 0
        n_true = 0
        n_pred = 0

        all_scores = []
        header = ['Total  ', 'Predictions', 'Correct', 'Precision', 'Recall  ', 'F1-Measure']
        res += ''.ljust(max_label_width + 2) + '  '.join(header) + '\n'
        head_width = [len(h) for h in header]

        exclude_list = ['None']
        #if "matres" in self.name: exclude_list.append('VAGUE')

        for label, width, display_label in zip(self.labels, label_widths, display_labels):
            if label not in exclude_list:
                total_count = self.total_truths.get(label, 0)
                pred_count = self.total_predictions.get(label, 0)

                n_true += total_count
                n_pred += pred_count

                correct_count = self.confusion_mat[label][label]
                n_correct += correct_count

                precision = safe_division(correct_count, pred_count)
                recall = safe_division(correct_count, total_count)
                f1_score = safe_division(2 * precision * recall, precision + recall)
                all_scores.append((precision, recall, f1_score))
                self.res_dict[label] = (f1_score, total_count)
                row = [total_count, pred_count, correct_count, precision, recall, f1_score]
                row = [num_to_str(cell).ljust(w) for cell, w in zip(row, head_width)]
                row.insert(0, display_label.rjust(max_label_width))
                res += '  '.join(row) + '\n'

        # weighing by the truth label's frequency
        label_weights = [safe_division(self.total_truths.get(label, 0), self.num_tests)
                         for label in self.labels if label not in exclude_list]
        weighted_scores = [(w * p, w * r, w * f) for w, (p, r, f) in zip(label_weights, all_scores)]

        assert len(label_weights) == len(weighted_scores)

        res += '\n'
        res += '  '.join(['Weighted Avg'.rjust(max_label_width),
                          ''.ljust(head_width[0]),
                          ''.ljust(head_width[1]),
                          ''.ljust(head_width[2]),
                          num_to_str(sum(p for p, _, _ in weighted_scores)).ljust(head_width[3]),
                          num_to_str(sum(r for _, r, _ in weighted_scores)).ljust(head_width[4]),
                          num_to_str(sum(f for _, _, f in weighted_scores)).ljust(head_width[5])])

        print(n_correct, n_pred, n_true)

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)

        res += f'\n Total Examples: {self.num_tests}'
        res += f'\n Overall Precision: {num_to_str(precision)}'
        res += f'\n Overall Recall: {num_to_str(recall)}'
        res += f'\n Overall F1: {num_to_str(f1_score)} '
        self.rel_f1 = f1_score
        return res