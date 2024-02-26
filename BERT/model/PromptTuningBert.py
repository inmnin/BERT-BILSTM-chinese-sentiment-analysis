from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch


class PromptTuningBert(BertForSequenceClassification):
    def __init__(self, config, tokenizer, prompt_nums):
        super(PromptTuningBert, self).__init__(config)
        self.prompt_embedding = torch.nn.Embedding(prompt_nums, config.hidden_size)
        self.tokenizer = tokenizer
        self.prompt_nums = prompt_nums
        # 扩展BERT的word embeddings层
        new_embeddings = torch.nn.Embedding(len(tokenizer), config.hidden_size)
        new_embeddings.weight.data[:self.bert.config.vocab_size] = self.bert.embeddings.word_embeddings.weight.data
        self.bert.embeddings.word_embeddings = new_embeddings

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.bert.embeddings.word_embeddings(input_ids)

        for i in range(self.prompt_nums):
            # 动态生成 [PROMPTn] 的名称
            prompt_token = f"[PROMPT{i + 1}]"
            prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_token)

            prompt_indices = (input_ids == prompt_token_id).nonzero(as_tuple=True)
            if len(prompt_indices[0]) == 0:  # 如果找不到 [PROMPTn] 标记，就中断
                break

            # 获取第 i 个 prompt 的 embedding
            prompt_embed = self.prompt_embedding(torch.tensor([i]).to(input_ids.device))

            # 更新 [PROMPTn] 标记的 embedding
            embeddings[prompt_indices[0], prompt_indices[1], :] += prompt_embed


        # #中途解码
        # # 使用词嵌入对embeddings进行解码
        # token_logits = torch.matmul(embeddings, self.bert.embeddings.word_embeddings.weight.T)
        # predicted_token_ids = token_logits.argmax(-1)
        #
        # # 仅解码前prompt_nums+2个词
        # decoded_sentence = self.tokenizer.decode(predicted_token_ids[0, :self.prompt_nums + 20].tolist())
        # print(f"Decoded sentence: {decoded_sentence}")


        sequence_output = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask).last_hidden_state

        # # 提取前 prompt_nums+1 个词的隐藏状态
        # relevant_hidden_states = sequence_output[:, :self.prompt_nums + 1]
        #
        # # 使用BERT tokenizer解码
        # decoded_tokens = self.tokenizer.decode(relevant_hidden_states[0].argmax(dim=-1))
        # print(f"Decoded tokens: {decoded_tokens}")


        return sequence_output


