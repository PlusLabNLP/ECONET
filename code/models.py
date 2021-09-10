from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel, BertConfig, RobertaConfig, RobertaModel

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_m\
odel.bin",
}

class TEClassifierRoberta(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, num_classes=5, finetune=True):
        super(TEClassifierRoberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.Tanh()
        self.num_classes = num_classes
        self.linear1_te = nn.Linear(config.hidden_size*2, mlp_hid)
        self.linear2_te = nn.Linear(mlp_hid, self.num_classes)

        self.init_weights()
        if not finetune:
            for name, param in self.roberta.named_parameters():
                param.requires_grad = False

    def forward(self, input_ids_te, token_type_ids_te=None, attention_mask_te=None,
                lidx_s=None, lidx_e=None, ridx_s=None, ridx_e=None, length_te=None,
                labels_te=None):

        batch_max = length_te.max()
        flat_input_ids_te = input_ids_te.view(-1, input_ids_te.size(-1))[:, :batch_max]
        flat_token_type_ids_te = token_type_ids_te.view(-1, token_type_ids_te.size(-1))[:, :batch_max]
        flat_attention_mask_te = attention_mask_te.view(-1, attention_mask_te.size(-1))[:, :batch_max]
        out, _ = self.roberta.forward(flat_input_ids_te,
                                       token_type_ids=flat_token_type_ids_te,
                                       attention_mask=flat_attention_mask_te)
        batch = out.size()[0]
        ltar_s = torch.cat([out[b, lidx_s[b], :] for b in range(batch)], dim=0).squeeze(1)
        rtar_s = torch.cat([out[b, ridx_s[b], :] for b in range(batch)], dim=0).squeeze(1)
        out = self.dropout(torch.cat([ltar_s, rtar_s], dim=1))
        # print(out.size())
        # linear prediction
        out = self.linear1_te(out)
        # print(out.size())
        out = self.act(out)
        out = self.linear2_te(out)
        # print(out.size())
        logits_te = out.view(-1, self.num_classes)

        if labels_te is not None:
            loss_fct = CrossEntropyLoss()
            loss_te = loss_fct(logits_te, labels_te)
            return loss_te, logits_te
        else:
            return logits_te


class TEClassifier(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"
    def __init__(self, config, mlp_hid=16, num_classes=5, finetune=True):
        super(TEClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.Tanh()
        self.num_classes = num_classes
        self.linear1_te = nn.Linear(config.hidden_size*2, mlp_hid)
        self.linear2_te = nn.Linear(mlp_hid, self.num_classes)

        self.init_weights()
        if not finetune:
            for name, param in self.roberta.named_parameters():
                param.requires_grad = False

    def forward(self, input_ids_te, token_type_ids_te=None, attention_mask_te=None,
                lidx_s=None, lidx_e=None, ridx_s=None, ridx_e=None, length_te=None,
                labels_te=None):

        batch_max = length_te.max()
        flat_input_ids_te = input_ids_te.view(-1, input_ids_te.size(-1))[:, :batch_max]
        flat_token_type_ids_te = token_type_ids_te.view(-1, token_type_ids_te.size(-1))[:, :batch_max]
        flat_attention_mask_te = attention_mask_te.view(-1, attention_mask_te.size(-1))[:, :batch_max]
        out, _ = self.bert.forward(flat_input_ids_te,
                                    token_type_ids=flat_token_type_ids_te,
                                    attention_mask=flat_attention_mask_te)
        batch = out.size()[0]
        ltar_s = torch.cat([out[b, lidx_s[b], :] for b in range(batch)], dim=0).squeeze(1)
        rtar_s = torch.cat([out[b, ridx_s[b], :] for b in range(batch)], dim=0).squeeze(1)
        out = self.dropout(torch.cat([ltar_s, rtar_s], dim=1))
        # print(out.size())
        # linear prediction
        out = self.linear1_te(out)
        # print(out.size())
        out = self.act(out)
        out = self.linear2_te(out)
        # print(out.size())
        logits_te = out.view(-1, self.num_classes)

        if labels_te is not None:
            loss_fct = CrossEntropyLoss()
            loss_te = loss_fct(logits_te, labels_te)
            return loss_te, logits_te
        else:
            return logits_te

class MultitaskClassifier(BertPreTrainedModel):
    def __init__(self, config, mlp_hid=16):
        super(MultitaskClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        outputs = self.dropout(outputs[0])
        # QA MLP                                                                
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear1(vectors))
        logits = self.linear2(outputs)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits

    
class MultitaskClassifierRoberta(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16):
        super(MultitaskClassifierRoberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear_2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, offsets, lengths, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        outputs = self.dropout(outputs[0])
        idx = 0
        vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                vectors.append(outputs[b, offsets[idx], :].unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        vectors = torch.cat(vectors, dim=0)
        outputs = self.act(self.linear_1(vectors))
        logits = self.linear_2(outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits


class RobertaQAClassifier(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16):
        super(RobertaQAClassifier, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        outputs = self.dropout(outputs[0]).reshape(-1, outputs[0].size()[-1])
        outputs = self.act(self.linear1(outputs))
        logits = self.linear2(outputs)

        return logits.reshape(input_ids.size()[0], input_ids.size()[1], -1)
