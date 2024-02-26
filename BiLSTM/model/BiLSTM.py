import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from BiLSTM.model.Selfattention_Head import SelfAttention

torch.manual_seed(123456)
class BiLSTM(nn.Module):
    """
        Implementation of BLSTM for sentiment classification task
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, tokenizer = None,max_len=150, dropout=0.5):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tokenizer = tokenizer

        # Sentence encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.attention_head = SelfAttention(hidden_dim)


    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    #删掉了padding的提示，效果可能不好

    def forward(self, sen_batch, sen_lengths):
        # batch_size = len(sen_batch)
        # Pack sequence
        packed_input = pack_padded_sequence(sen_batch, sen_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.sen_rnn(packed_input)

        # Unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        fw_output = output[:, :, :self.hidden_dim]
        bw_output = output[:, :, self.hidden_dim:]
        output = fw_output + bw_output

        output = self.attention_head(output)
        # Average pooling
        # sum_hidden_fw = torch.sum(output[:, :, :self.hidden_dim], dim=1)
        # avg_hidden_fw = sum_hidden_fw / sen_lengths.unsqueeze(1)
        #
        # sum_hidden_bw = torch.sum(output[:, :, self.hidden_dim:], dim=1)
        # avg_hidden_bw = sum_hidden_bw / sen_lengths.unsqueeze(1)
        #
        # avg_hidden = torch.cat([avg_hidden_fw, avg_hidden_bw], dim=1)

        return output

class BI(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, block_nums, tokenizer=None, max_len=150, dropout=0.5):
        super(BI, self).__init__()

        self.layers = nn.ModuleList([
            BiLSTM(input_dim,  # 双向的，所以输出是 2 * hidden_dim
                   hidden_dim,
                   num_layers,
                   output_dim,
                   tokenizer,
                   max_len,
                   dropout)
            for i in range(block_nums)
        ])

        self.classifation = nn.Linear(hidden_dim,output_dim)
    def forward(self, x, lengths):
        for layer in self.layers:
            x = layer(x, lengths)
            # 对seq_length维度进行平均
        x = torch.mean(x, dim=1)
        x = self.classifation(x)
        return x










class BiLSTM_PAL(nn.Module):
    """
        Implementation of BLSTM for sentiment classification task
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,max_len=150, dropout=0.5):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.classification = nn.Linear(2*hidden_dim,output_dim)
        self.Softmax = nn.Softmax()

        # Sentence encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)


    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    #删掉了padding的提示，效果可能不好
    def forward(self, sen_batch, sen_lengths):
        batch_size = len(sen_batch)
        sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)

        # 提取每个句子的 [CLS] 标记的隐藏状态
        #有待修改，应该对所有token对应的隐藏状态采取池化策略
        cls_hidden_states = sen_rnn[:,  sen_lengths:]
        return cls_hidden_states


