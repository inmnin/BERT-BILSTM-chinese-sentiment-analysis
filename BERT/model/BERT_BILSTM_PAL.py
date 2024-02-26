import torch.nn.functional as F
import torch.nn as nn
import torch

from BERT.model.PromptTuningBert import PromptTuningBert
from BERT.model.PromptTunningBert_PAL import PromptTuningBert_PAL
from BERT.model.Base_Bert import Base_Bert
from BiLSTM.model.BiLSTM import BiLSTM
from BiLSTM.model.BiLSTM_cls import BiLSTM_cls
from transformers import BertForSequenceClassification


class BERT_BiLSTM_PAL(nn.Module):
    def __init__(self, bert:BertForSequenceClassification, bilstm:BiLSTM_cls):
        super(BERT_BiLSTM_PAL, self).__init__()  # 这一行应该在其他初始化语句之前
        self.bert = bert
        self.bilstm = bilstm
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.classifation = nn.Linear(2*self.bilstm.hidden_dim,2)
        self.bert_weight = nn.Parameter(torch.tensor(1.0))
        self.bi_weight = nn.Parameter(torch.tensor(1.0))
        # self.bert_weight = 4
        # self.bi_weight = 1

    def forward(self, bert_input_ids,bi_input_ids, bert_attention_mask, bi_attention_mask,labels):

        sum_weight = self.bi_weight+self.bert_weight
        bert_out = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask, labels=labels)
        bert_out = bert_out.logits
        # bert_out = self.softmax(bert_tmp.logits)
        bert_out = self.bert_weight/sum_weight * bert_out

        # Since all sequences are of length 150
        #这里用两个[CLS]算了句子长度
        actual_lengths = bi_attention_mask.sum(dim=1)

        with torch.no_grad():
            bi_embeddings = self.bert.bert.embeddings(bi_input_ids)
        # Pass the BERT output directly through BiLSTM
        bilstm_out = self.bilstm(bi_embeddings, actual_lengths)
        # bilstm_out = self.classifation(bilstm_out)
        # bilstm_out = self.softmax(bilstm_out)
        bilstm_out = self.bi_weight / sum_weight * bilstm_out
        return bilstm_out+bert_out
        # return bilstm_out