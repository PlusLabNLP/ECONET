import logging
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertConfig, RobertaConfig, RobertaModel, \
    ElectraConfig, ElectraModel,XLNetConfig, XLNetModel, LongformerConfig, LongformerModel


logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

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

class RobertaRelationClassifier(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16):
        super(RobertaRelationClassifier, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = 3
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size*2, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)

        # For QA, it's a binary classifier
        self.linear1_qa = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_qa = nn.Linear(mlp_hid, 2)

        # For event, it's also a binary classifier
        self.linear1_event = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_event = nn.Linear(mlp_hid, 2)

        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, query_events, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, end_to_end=False):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        query = torch.gather(outputs[0], 1, query_events.view(-1, 1).unsqueeze(2).expand_as(outputs[0]))

        query_append_events = torch.cat((outputs[0], query), dim=2)

        IE_out = self.dropout(query_append_events).reshape(-1, query_append_events.size()[-1])
        IE_out = self.act(self.linear1(IE_out))
        IE_logits = self.linear2(IE_out).reshape(input_ids.size()[0], input_ids.size()[1], -1)

        outputs = self.dropout(outputs[0]).reshape(-1, outputs[0].size()[-1])
        outputs_qa = self.act(self.linear1_qa(outputs))
        QA_logits = self.linear2_qa(outputs_qa).reshape(input_ids.size()[0], input_ids.size()[1], -1)

        logits = (IE_logits, QA_logits)

        if end_to_end:
            outputs = self.act(self.linear1_event(outputs))
            event_logits = self.linear2_event(outputs).reshape(input_ids.size()[0], input_ids.size()[1], -1)
            logits += (event_logits,)

        return logits


class RobertaEventClassifier(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16):
        super(RobertaEventClassifier, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # For event, it's also a binary classifier
        self.linear1_event = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_event = nn.Linear(mlp_hid, 2)

        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        outputs = self.dropout(outputs[0]).reshape(-1, outputs[0].size()[-1])
        outputs = self.act(self.linear1_event(outputs))
        logits = self.linear2_event(outputs).reshape(input_ids.size()[0], input_ids.size()[1], -1)

        return logits

class ElectraEventClassifier(BertPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"
    def __init__(self, config, mlp_hid=16):
        super(ElectraEventClassifier, self).__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # For event, it's also a binary classifier
        self.linear1_event = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_event = nn.Linear(mlp_hid, 2)

        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        outputs = self.electra(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        outputs = self.dropout(outputs[0]).reshape(-1, outputs[0].size()[-1])
        outputs = self.act(self.linear1_event(outputs))
        logits = self.linear2_event(outputs).reshape(input_ids.size()[0], input_ids.size()[1], -1)

        return logits


class RobertaTemporalLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, num_classes=2):
        super(RobertaTemporalLM, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = num_classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, offsets, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids)

        masked_token = torch.gather(outputs[0], 1, offsets.view(-1, 1).unsqueeze(2).expand_as(outputs[0]))[:, 0, :]
        #logger.info(masked_token.size())
        out = self.dropout(masked_token)
        out = self.act(self.linear1(out))
        #logger.info(out.size())
        logits = self.linear2(out).reshape(input_ids.size()[0], -1)
        #logger.info(logits.size())
        return logits

class RobertaJointLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, num_classes_tmp=2, num_classes_evt=2):
        super(RobertaJointLM, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1_tmp = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_tmp = nn.Linear(mlp_hid, num_classes_tmp)

        self.linear1_evt = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_evt = nn.Linear(mlp_hid, num_classes_evt)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids=None, offsets=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               inputs_embeds=inputs_embeds)
        masked_token = torch.gather(outputs[0], 1, offsets.view(-1, 1).unsqueeze(2).expand_as(outputs[0]))[:, 0, :]
        out = self.act(self.linear1_tmp(self.dropout(masked_token)))
        logits_tmp = self.linear2_tmp(out).reshape(offsets.size()[0], -1)

        out = self.act(self.linear1_evt(self.dropout(masked_token)))
        logits_evt = self.linear2_evt(out).reshape(offsets.size()[0], -1)
        # if regular branch, return emb for calculating perturbed embeddings
        if inputs_embeds == None:
            inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)
        return logits_tmp, logits_evt, inputs_embeds

class BertJointLM(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"
    def __init__(self, config, mlp_hid=16, num_classes_tmp=2, num_classes_evt=2):
        super(BertJointLM, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1_tmp = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_tmp = nn.Linear(mlp_hid, num_classes_tmp)

        self.linear1_evt = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_evt = nn.Linear(mlp_hid, num_classes_evt)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids=None, offsets=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None):
        outputs = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               inputs_embeds=inputs_embeds)
        masked_token = torch.gather(outputs[0], 1, offsets.view(-1, 1).unsqueeze(2).expand_as(outputs[0]))[:, 0, :]
        out = self.act(self.linear1_tmp(self.dropout(masked_token)))
        logits_tmp = self.linear2_tmp(out).reshape(offsets.size()[0], -1)

        out = self.act(self.linear1_evt(self.dropout(masked_token)))
        logits_evt = self.linear2_evt(out).reshape(offsets.size()[0], -1)
        # if regular branch, return emb for calculating perturbed embeddings
        if inputs_embeds == None:
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        return logits_tmp, logits_evt, inputs_embeds


class RobertaJointLM_MaskAll(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, num_classes_tmp=2, num_classes_evt=2):
        super(RobertaJointLM_MaskAll, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1_tmp = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_tmp = nn.Linear(mlp_hid, num_classes_tmp)

        self.linear1_evt = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_evt = nn.Linear(mlp_hid, num_classes_evt)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids)
        outputs = outputs[0]
        outputs = outputs.reshape(-1, outputs.size()[-1])

        out = self.act(self.linear1_tmp(self.dropout(outputs)))
        logits_tmp = self.linear2_tmp(out).reshape(input_ids.size()[0], input_ids.size()[1], -1)

        out = self.act(self.linear1_evt(self.dropout(outputs)))
        logits_evt = self.linear2_evt(out).reshape(input_ids.size()[0], input_ids.size()[1], -1)

        return logits_tmp, logits_evt


class ContrastiveLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16, num_classes_tmp=2, num_classes_evt=2):
        super(ContrastiveLM, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1_tmp = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_tmp = nn.Linear(mlp_hid, num_classes_tmp)

        self.linear1_evt = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_evt = nn.Linear(mlp_hid, num_classes_evt)

        self.linear1_ctmp = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_ctmp = nn.Linear(mlp_hid, 2)

        self.linear1_cevt = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_cevt = nn.Linear(mlp_hid, 2)

        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids=None, offsets=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, task='mask'):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               inputs_embeds=inputs_embeds)

        masked_token = torch.gather(outputs[0], 1, offsets.view(-1, 1).unsqueeze(2).expand_as(outputs[0]))[:, 0, :]
        if task == 'mask':
            out = self.act(self.linear1_tmp(self.dropout(masked_token)))
            logits_tmp = self.linear2_tmp(out).reshape(offsets.size()[0], -1)

            out = self.act(self.linear1_evt(self.dropout(masked_token)))
            logits_evt = self.linear2_evt(out).reshape(offsets.size()[0], -1)
            if inputs_embeds == None:
                inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)
            return logits_tmp, logits_evt, inputs_embeds
        elif task == 'contrastive':
            out = self.act(self.linear1_ctmp(self.dropout(masked_token)))
            logits_ct = self.linear2_ctmp(out).reshape(input_ids.size()[0], -1)

            out = self.act(self.linear1_cevt(self.dropout(masked_token)))
            logits_ce = self.linear2_cevt(out).reshape(input_ids.size()[0], -1)
            return logits_ct, logits_ce


class ContrastiveLM_Bert(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"
    def __init__(self, config, mlp_hid=16, num_classes_tmp=2, num_classes_evt=2):
        super(ContrastiveLM_Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1_tmp = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_tmp = nn.Linear(mlp_hid, num_classes_tmp)

        self.linear1_evt = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_evt = nn.Linear(mlp_hid, num_classes_evt)

        self.linear1_ctmp = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_ctmp = nn.Linear(mlp_hid, 2)

        self.linear1_cevt = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2_cevt = nn.Linear(mlp_hid, 2)

        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids=None, offsets=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, task='mask'):
        outputs = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               inputs_embeds=inputs_embeds)

        masked_token = torch.gather(outputs[0], 1, offsets.view(-1, 1).unsqueeze(2).expand_as(outputs[0]))[:, 0, :]
        if task == 'mask':
            out = self.act(self.linear1_tmp(self.dropout(masked_token)))
            logits_tmp = self.linear2_tmp(out).reshape(offsets.size()[0], -1)

            out = self.act(self.linear1_evt(self.dropout(masked_token)))
            logits_evt = self.linear2_evt(out).reshape(offsets.size()[0], -1)
            if inputs_embeds == None:
                inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
            return logits_tmp, logits_evt, inputs_embeds
        elif task == 'contrastive':
            out = self.act(self.linear1_ctmp(self.dropout(masked_token)))
            logits_ct = self.linear2_ctmp(out).reshape(input_ids.size()[0], -1)

            out = self.act(self.linear1_cevt(self.dropout(masked_token)))
            logits_ce = self.linear2_cevt(out).reshape(input_ids.size()[0], -1)
            return logits_ct, logits_ce