from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
from transformers import BertConfig


class Base_Bert(BertForSequenceClassification):
    def __init__(self, config, tokenizer):
        super(Base_Bert, self).__init__(config)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, labels=None):
        # embeddings = self.bert.embeddings.word_embeddings(input_ids)
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return sequence_output