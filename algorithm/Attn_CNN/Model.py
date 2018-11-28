import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.Attn import Attn
from algorithm.Embedding import Embedding


class Attn_CNN(nn.Module):
    '''
    Attention CNN模型
    1. 将index转换为embedding
    2. 计算每个word与relation embedding之间的attention权重
    3.. 使用3*hidden_size的卷积核在带权的sentence embedding上进行卷积
    4. 进行max pooling
    5. 与relation embedding 拼接后喂入全连接
    6. 返回out
    '''

    def __init__(self, hidden_size, out_size, voc):
        super(Attn_CNN, self).__init__()

        # self.embedding = Embedding(hidden_size, voc, dropout=0.1)
        pe_size = 5
        self.word_embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        self.pos_embedding = torch.nn.Embedding(300 * 2, pe_size)


        Ci = 1
        Co = 16
        k = 3
        D = hidden_size + 2 * pe_size
        self.attn = Attn("general", hidden_size, D)

        self.conv = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(k, D))

        self.out = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(Co + hidden_size, out_size)
        )

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, position_to_entity1_batch,
                position_to_entity2_batch, hidden=None):
        # embedded_sentences = self.embedding(sentences_seq=sentences_seq,
        #                                     position_to_entity1_batch=position_to_entity1_batch,
        #                                     position_to_entity2_batch=position_to_entity2_batch)  # N*W*D
        # embedded_entity1 = self.embedding(entity_index=entity1_index)
        # embedded_entity2 = self.embedding(entity_index=entity2_index)
        embedded_sentences = self.word_embedding(sentences_seq.t())
        pe1 = self.pos_embedding(position_to_entity1_batch.t())
        pe2 = self.pos_embedding(position_to_entity2_batch.t())
        embedded_sentences = torch.cat((embedded_sentences, pe1, pe2), dim=2)

        embedded_entity1 = self.word_embedding(entity1_index)
        embedded_entity2 = self.word_embedding(entity2_index)

        embedded_relation = (embedded_entity1 - embedded_entity2).unsqueeze(1)  # N*1*D

        attn_weights = self.attn(embedded_relation,embedded_sentences)

        weighted_sentence = attn_weights.transpose(1, 2) * embedded_sentences

        # 将句子向量维度变成 N*Ci*W*D
        # N为batch_size
        # Ci为通道数，文本为1
        # W为句子长度
        # D为词向量长度
        weighted_sentence = weighted_sentence.unsqueeze(1)

        x = self.conv(weighted_sentence)  # N*Co*W*1

        x = F.relu(x).squeeze(3)  # N*Co*W

        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # N*Co

        out = self.out(torch.cat((x, embedded_relation.squeeze(1)), dim=1))
        return out
