from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn as nn


class PromptTuningBert_PAL(BertForSequenceClassification):
    def __init__(self, config, tokenizer):
        super(PromptTuningBert_PAL, self).__init__(config)
        self.tokenizer = tokenizer
        self.classification = nn.Linear(768,2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.bert.embeddings.word_embeddings(input_ids)

        # 计算BERT的输出
        sequence_output = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask).last_hidden_state

        # 获取每个句子中[MASK]标记的位置
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero().squeeze(-1)
        # 从sequence_output中获取每个句子中[MASK]标记的隐藏状态
        mask_hidden_states = sequence_output[torch.arange(sequence_output.size(0)), mask_positions]
        logits = self.projection(mask_hidden_states)
        return logits


