import torch.nn as nn
import copy, math
from algorithm.Attn import Attn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.attention_func = Attn("scaled", None, None)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention_func(query, key, value, mask=mask,
                                           dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        # return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, m, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, m, m, mask))
        return self.sublayer[1](x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, hidden_size, out_size, voc):
        super(Transformer, self).__init__()

        c = copy.deepcopy
        dropout = 0.1
        h = 1
        N = 1
        attn = MultiHeadedAttention(h, hidden_size)
        ff = PositionwiseFeedForward(hidden_size, hidden_size * 4, dropout)
        position = PositionalEncoding(hidden_size, dropout)

        # 构建encoder层
        encode_layer = EncoderLayer(hidden_size, c(attn), c(ff), dropout)
        self.encode_layers = clones(encode_layer, N)

        # 构建decoder层
        self.decode_layers = clones(DecoderLayer(hidden_size, c(attn), c(ff), dropout), N)

        self.norm = LayerNorm(encode_layer.size)
        self.embeddings = torch.nn.Embedding(voc.num_words, hidden_size)
        self.sentence_embeddings = nn.Sequential(self.embeddings, c(position))
        self.out = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(hidden_size, out_size)
        )

    def encode(self, x, mask):
        for layer in self.encode_layers:
            x = layer(x, mask)
        return self.norm(x)

    def decode(self, x, m, mask):
        for layer in self.decode_layers:
            x = layer(x, m, mask)
        return self.norm(x)

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, position_to_entity1_batch,
                position_to_entity2_batch):
        # 将输入转化成embedding
        sentences_embedding = self.sentence_embeddings(sentences_seq.t())
        entity1_embedding = self.embeddings(entity1_index)
        entity2_embedding = self.embeddings(entity2_index)
        relation_embedding = entity1_embedding - entity2_embedding

        # 获取输入的mask
        mask = None

        x = self.encode(sentences_embedding, mask)  # batch * length * embedding_size
        x = self.decode(relation_embedding, x, mask)

        out = self.out(x[:, -1])

        return out
