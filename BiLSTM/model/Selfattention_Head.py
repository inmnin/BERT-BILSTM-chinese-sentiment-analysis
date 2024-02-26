import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x, mask=None):
        N = x.shape[0]
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # Scaled dot-product attention
        energy = torch.bmm(queries, keys.transpose(1, 2))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** 0.5), dim=2)
        out = torch.bmm(attention, values)
        return out

# In your BiLSTM_cls:
# self.attention_head = SelfAttention(hidden_dim)
