import torch
import torch.nn as nn
from algorithm.Attn import Attn

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class AttnGRU(nn.Module):
    '''
    1. 将word index转化为 embedding
    2. 计算relation embedding
    3. 计算每个word与relation embedding之间的attention权重
    4. 计算加权后的embedding，并进行pack padding.
    5. 计算bidirectional GRU结果的和为outputs
    6. 将outputs最后一个维度与relation embedding进行concat，喂入全连接，得到out
    6. 返回out
    '''

    def __init__(self, hidden_size, out_size, voc, device, n_layers=1, dropout=0):
        super(AttnGRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        self.attn = Attn("general", hidden_size).to(device)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True, batch_first=True)
        self.out = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, out_size)
        )

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, position_to_entity1_batch,
                position_to_entity2_batch, hidden=None):
        # 将句子与entity进行embedding
        embedded_sentences = self.embedding(sentences_seq.t())
        embedded_entity1 = self.embedding(entity1_index)
        embedded_entity2 = self.embedding(entity2_index)

        # 计算relation embedding
        embedded_relation = (embedded_entity1 - embedded_entity2).unsqueeze(1)
        attn_weights = self.attn(embedded_sentences, embedded_relation)

        weighted_sentence = attn_weights.transpose(1, 2) * embedded_sentences

        # 使用pack_padded_sequence对边长序列进行pack
        packed = torch.nn.utils.rnn.pack_padded_sequence(weighted_sentence, sentence_lengths, batch_first=True)

        self.gru.flatten_parameters()
        outputs, hidden = self.gru(packed, hidden)

        # unpack outputs
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # 将双向的两个向量求和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        out = self.out(torch.cat((outputs[:, -1, :], embedded_relation.squeeze(1)), dim=1))

        return out
