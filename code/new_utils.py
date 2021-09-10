import logging
import numpy as np
import torch

logging.basicConfig(format='%(asctime)ss - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

labels_to_idx = {'before': 0, 'after': 1, 'during': 2}
idx_to_labels = {0: 'before', 1: 'after', 2: 'during'}
reverse_map = {'before': 'after', 'after': 'before', 'during': 'during'}


def compute_loss(logits, labels, loss_fct, device):
    logits = torch.cat(logits, dim=0)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return loss_fct(logits, labels)

def exact_match(question_ids, labels, predictions):
    em = {}
    for q, l, p in zip(question_ids, labels, predictions):
        em[q] = int(l == p)
    return sum(em.values()) / float(len(em))

def predict_events(logits, offsets):
    batch_event_pred = []
    filtered_logits = []
    for b, offset in enumerate(offsets):
        event_pred = []
        for i in range(len(offset)):
            event_pred.append(np.argmax(logits[b, offset[i], :].detach().cpu().numpy()))
            filtered_logits.append(logits[b, offset[i], :].unsqueeze(0))
        batch_event_pred.append(event_pred)
    return batch_event_pred, filtered_logits

def filter_outputs(logits, query_events, offsets, events, labels, temp_rels):
    filtered_logits_pos, filtered_logits_neg = [], []
    new_labels_pos, new_labels_neg = [], []
    for b, offset in enumerate(offsets):
        query_event_offset = query_events[b]
        for i in range(len(offset)):
            if events[b][i] == 1 and offset[i] != query_event_offset:
                if labels[b][i] == 1:
                    filtered_logits_pos.append(logits[b, offset[i], :].unsqueeze(0))
                    new_labels_pos.append(labels_to_idx[temp_rels[b]])
                else:
                    filtered_logits_neg.append(logits[b, offset[i], :].unsqueeze(0))
                    new_labels_neg.append(labels_to_idx[temp_rels[b]])

    return filtered_logits_pos, filtered_logits_neg, new_labels_pos, new_labels_neg


def filter_outputs_qa(logits, offsets, labels, events):

    filtered_logits = []
    filtered_labels = []
    for b, offset in enumerate(offsets):
        assert len(events[b]) == len(offset)
        for i in range(len(offset)):
            if events[b][i] == 1:
                filtered_logits.append(logits[b, offset[i], :].unsqueeze(0))
                filtered_labels.append(labels[b][i])

    return filtered_logits, filtered_labels

def batch_predict_joint(IE_logits, QA_logits, query_events, offsets, events, temp_rels, iw=1.0, qw=1.0):
    batch_preds = []
    for b, offset in enumerate(offsets):
        query_event_offset = query_events[b]
        preds = []
        for i in range(len(offset)):
            if events[b][i] == 1:
                if offset[i] != query_event_offset:
                    prob_ie = torch.nn.functional.softmax(IE_logits[b, offset[i], :]).detach().cpu().tolist()
                    prob_qa = torch.nn.functional.softmax(QA_logits[b, offset[i], :]).detach().cpu().tolist()
                    prob_pos = iw*prob_ie[labels_to_idx[temp_rels[b]]] + qw*prob_qa[1]
                    prob_neg = iw*max([prob_ie[i] for i in range(3)
                                       if i != labels_to_idx[temp_rels[b]]]) + qw*prob_qa[0]
                    ind = 1 if prob_pos > prob_neg else 0
                else:
                    #ind = 0
                    ind = np.argmax(QA_logits[b, offset[i], :].detach().cpu().numpy())
                preds.append(ind)
            else:
                preds.append(0)
        batch_preds.append(preds)
    return batch_preds

def batch_predict(IE_logits, query_events, offsets, events, temp_rels):
    batch_preds = []
    for b, offset in enumerate(offsets):
        query_event_offset = query_events[b]
        preds = []
        for i in range(len(offset)):
            if events[b][i] == 1:
                if offset[i] != query_event_offset:
                    pred = np.argmax(IE_logits[b, offset[i], :].detach().cpu().numpy())
                    pred = idx_to_labels[pred]
                    ind = 1 if pred == temp_rels[b] else 0
                else:
                    ind = 0
                preds.append(ind)
            else:
                preds.append(0)
        batch_preds.append(preds)
    return batch_preds


def batch_predict_qa(logits, offsets, events):
    batch_preds = []
    for b, offset in enumerate(offsets):
        preds = []
        assert len(events[b]) == len(offset)
        for i in range(len(offset)):
            if events[b][i] == 1:
                pred = np.argmax(logits[b, offset[i], :].detach().cpu().numpy())
                preds.append(int(pred))
            else:
                preds.append(0)
        assert len(preds) == len(events[b])
        batch_preds.append(preds)
    return batch_preds

def convert_to_features_roberta_question_seg(data, tokenizer, max_length=150, evaluation=False):
    # each sample will have <s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating

    for k, v in data.items():
        new_tokens = ["<s>"]
        orig_to_tok_map = []
        has_rel, has_evt = False, False
        for i, token in enumerate(v['question']):
            if v['question_segments'][i] == 1 and not has_rel:
                rel_idx = i
                has_rel = True
            if v['question_segments'][i] == 2 and not has_evt:
                evt_idx = i
                has_evt = True

            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("</s>")

        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['question']) + len(v['context']) + 1 # account for </s>

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

        labels, offsets = [], []
        for kk, vv in enumerate(v['answers']['labels']):
            labels.append(vv)
            offsets.append(orig_to_tok_map[kk + len(v['question'])])

        sample = {'label': labels,
                  'offset': offsets,
                  'events': v['answers']['types'],
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k}

        assert has_rel == has_evt
        if has_rel:
            sample['temp_rel'] = orig_to_tok_map[rel_idx]
            sample['temp_rel_mask'] = 1
        else:
            sample['temp_rel'] = 0
            sample['temp_rel_mask'] = 0
        if has_evt:
            sample['query_event'] = orig_to_tok_map[evt_idx]
        else:
            sample['query_event'] = 0

        # add these three field for qualitative analysis
        if evaluation:
            sample['passage'] = v['context']
            sample['question'] = v['question']
            sample['question_cluster'] = v['question_cluster']
            sample['cluster_size'] = v['cluster_size']
            sample['answer'] = v['answers']
            sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
        samples.append(sample)

        # check some example data
        if counter < 0:
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples


def convert_to_features_roberta_qa(data, tokenizer, max_length=150, evaluation=False, ie=False):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0  # to show global max_len without truncating

    for k, v in data.items():
        segment_ids = []
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])

        new_tokens = ["</s>", "</s>"]  # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1  # account for ending </s>

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
        # duplicate P + Q for each answer
        labels, offsets = [], []
        for kk, vv in enumerate(v['answers']['labels']):
            labels.append(vv)
            offsets.append(orig_to_tok_map[kk] + len(question) + 1)

        sample = {'label': labels,
                  'events': v['answers']['types'],
                  'offset': offsets,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k}
        if ie:
            sample['temp_rel'] = v['temp_rel']
            sample['query_event'] = orig_to_tok_map[v['question_event_idx']] + len(question) + 1

        # add these three field for qualitative analysis
        if evaluation:
            sample['passage'] = v['context']
            sample['question'] = v['question']
            sample['question_cluster'] = v['question_cluster']
            sample['cluster_size'] = v['cluster_size']
            sample['answer'] = v['answers']
            sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
        samples.append(sample)

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples