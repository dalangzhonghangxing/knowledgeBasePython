import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    '''
    双向LSTM模型
    1. 将index转换为embedding
    2. 将embedded_sentence喂入lstm
    3. 将双向lstm的结果求和
    4. 将lstm的最后一层输出与embedded_relation进行concat，喂入全连接
    5. 返回out
    '''

    def __init__(self, hidden_size, out_size, voc, dropout=0):
        super(CNN, self).__init__()

        self.embedding = torch.nn.Embedding(voc.num_words, hidden_size)

        Ci = 1
        Co = 16
        k = 3
        D = hidden_size

        self.conv = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(k, D))

        self.out = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(Co + D, out_size)
        )

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, hidden=None):
        embedded_sentences = self.embedding(sentences_seq.t())  # N*W*D
        embedded_entity1 = self.embedding(entity1_index)
        embedded_entity2 = self.embedding(entity2_index)

        # 将句子向量维度变成 N*Ci*W*D
        # N为batch_size
        # Ci为通道数，文本为1
        # W为句子长度
        # D为词向量长度
        embedded_sentences = embedded_sentences.unsqueeze(1)

        x = self.conv(embedded_sentences)  # N*Co*W*1

        x = F.relu(x).squeeze(3)  # N*Co*W

        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # N*Co

        embedded_relation = embedded_entity1 - embedded_entity2  # N*1*D

        out = self.out(torch.cat((x, embedded_relation), dim=1))
        return out
