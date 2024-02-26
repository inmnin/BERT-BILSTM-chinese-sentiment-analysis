import torch.nn.functional as F
import torch.nn as nn
import torch

from BERT.model.PromptTuningBert import PromptTuningBert
from BiLSTM.model.BiLSTM import BiLSTM
from BERT.model.Base_Bert import Base_Bert

class BERT_BiLSTM(nn.Module):
    def __init__(self, bert:Base_Bert, bilstm:BiLSTM):
        super(BERT_BiLSTM, self).__init__()  # 这一行应该在其他初始化语句之前
        self.bert = bert
        self.bilstm = bilstm
        self.loss_fn = nn.CrossEntropyLoss()
        self.classification = nn.Linear(2*self.bilstm.hidden_dim, 2)
        self.softmax = nn.Softmax()

    def forward(self, input_ids, attention_mask, labels):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Since all sequences are of length 150
        actual_lengths = attention_mask.sum(dim=1)

        # Pass the BERT output directly through BiLSTM
        bilstm_out_prob = self.bilstm(bert_out, actual_lengths)
        tmp_out = self.classification(bilstm_out_prob)
        out = self.softmax(tmp_out)
        return out