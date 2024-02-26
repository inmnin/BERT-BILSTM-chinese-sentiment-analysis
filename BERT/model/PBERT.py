from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn as nn

class PromptTuningBert(BertForSequenceClassification):
    def __init__(self, config,tokenizer):
        super(PromptTuningBert, self).__init__(config)
        self.classifier = nn.Linear(768,2)
        self.tokenizer = tokenizer
    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.bert.embeddings.word_embeddings(input_ids)
        sequence_output = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask).last_hidden_state
        # 获取 [MASK] 标记的位置
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero().squeeze(-1)
        # 提取 [MASK] 标记对应的隐藏状态
        # mask_hidden_states = sequence_output[mask_positions]
        mask_hidden_states = sequence_output[torch.arange(input_ids.size(0)), mask_positions[:, 1]]
        sequence_output = self.classifier(mask_hidden_states)
        return sequence_output


