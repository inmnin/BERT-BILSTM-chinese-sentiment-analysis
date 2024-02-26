import tqdm
from torch.utils.data import Dataset
import string
import torch
from transformers import BertTokenizer
from time import sleep
class Normal_dataset(Dataset):
    def __init__(self, corpus_path, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.contents = []
        self.labels = []

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.on_memory == True:
                self.lines = [line[:-1]
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                for line in self.lines:
                    content,label = self.split_at_last_colon(line)
                    if label == '' or content == '':
                        continue

                    self.contents.append(content)
                   #标签格式调试
                    if(str(label) != '0' and str(label) != '-1' and str(label) != '1'):
                        continue

                    self.labels.append(label) #可能出现字符串无法转换成整数的情况

    def __getitem__(self, index):
        content = self.contents[index]
        label = self.labels[index]

        # Convert content to a sequence of token indices
        #这里的句子是已经分词好了的
        content = content.split()
        return content, label

    @staticmethod
    def split_at_last_colon(s):
        idx = s.rfind(':')
        if idx == -1:  # no comma found
            return s, ''
        return s[:idx], s[idx + 1:]

    def remove_punctuation(self,s):
        """
        Remove all punctuation from a string.
        """
        return ''.join(ch for ch in s if ch not in string.punctuation)

    # def prepare_sequence(self,words):
    #     """
    #     Convert words to indices and adjust the sequence length according to seq_len.
    #     """
    #     idxs = [self.vocab.stoi[word] if word in self.vocab.stoi else self.vocab.unk_index for word in words]
    #
    #     if len(idxs) < self.seq_len:
    #         # Padding
    #         idxs += [self.vocab.pad_index] * (self.seq_len - len(idxs))
    #     elif len(idxs) > self.seq_len:
    #         # Truncating
    #         idxs = idxs[:self.seq_len]

        return idxs

    def __len__(self):
        return len(self.contents)


#BERT的dataset
class BertDataset(Normal_dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        super().__init__(corpus_path, seq_len, encoding, corpus_lines, on_memory)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        content = self.contents[index]
        label = int(self.labels[index])

        # 使用tokenizer对内容进行编码
        inputs = self.tokenizer(content, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=self.seq_len)

        inputs['original_text'] = self.contents[index]
        # 删除不必要的批次维度
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):  # 检查是否为张量
                inputs[key] = value.squeeze(0)

        inputs['labels'] = torch.tensor(label)
        return inputs


class PromptBertDataset(BertDataset):
    def __init__(self, corpus_path, tokenizer, seq_len, prompt_nums=0, encoding="utf-8", corpus_lines=None, on_memory=True):
        super().__init__(corpus_path, tokenizer, seq_len, encoding, corpus_lines, on_memory)
        self.prompt_nums = prompt_nums

    def __getitem__(self, index):
        content = ''
        for i in range(self.prompt_nums):
            prompt_token = f"[PROMPT{i + 1}]"
            content += prompt_token+' '
        content = content + "[CLS] "+self.contents[index]  # 在这里加入[PROMPT]


        # print(content)
        label = int(self.labels[index])
        # 使用tokenizer对内容进行编码
        inputs = self.tokenizer(content, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=self.seq_len + self.prompt_nums)  # 考虑到加入了一个额外的token，所以max_length增加1
        # 删除不必要的批次维度
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        inputs['original_text'] = self.contents[index]
        inputs['labels'] = torch.tensor(label)
        return inputs

class ParallelDataset(BertDataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        super().__init__(corpus_path, tokenizer, seq_len, encoding, corpus_lines, on_memory)
        self.prompt_nums=8
    def __getitem__(self, index):
        content = "这是一条[MASK]评论[SEP]" + self.contents[index] # 在这里加入[PROMPT]


        # print(content)
        label = int(self.labels[index])
        # 使用tokenizer对内容进行编码
        inputs = self.tokenizer(content, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=self.seq_len + self.prompt_nums)  # 考虑到加入了一个额外的token，所以max_length增加1
        # 删除不必要的批次维度
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)

        inputs['original_text'] = self.contents[index]
        inputs['labels'] = torch.tensor(label)
        return inputs
