import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    '''
    CNN模型
    1. 将index转换为embedding
    2. 使用3*hidden_size的卷积核进行卷积
    3. 进行max pooling
    4. 与relation embedding 拼接后喂入全连接
    5. 返回out
    '''

    def __init__(self, hidden_size, out_size, voc):
        super(CNN, self).__init__()

        # self.embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        pe_size = 10
        self.word_embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        self.pos_embedding = torch.nn.Embedding(300 * 2, pe_size)

        Ci = 1
        Co = 16
        D = hidden_size + 2 * pe_size

        self.conv2 = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(2, D), padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(3, D), padding=(1, 0))
        self.conv5 = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(5, D), padding=(2, 0))

        self.out = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Co * 3, out_size)
        )

    def conv(self, input, conv_func):
        x = conv_func(input)  # N*Co*W*1

        x = F.relu(x).squeeze(3)  # N*Co*W

        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # N*Co

        return x

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, position_to_entity1_batch,
                position_to_entity2_batch, hidden=None):
        embedded_sentences = self.word_embedding(sentences_seq.t())  # N*W*D
        # embedded_entity1 = self.word_embedding(entity1_index)
        # embedded_entity2 = self.word_embedding(entity2_index)
        pe1 = self.pos_embedding(position_to_entity1_batch.t())
        pe2 = self.pos_embedding(position_to_entity2_batch.t())
        embedded_sentences = torch.cat((embedded_sentences, pe1, pe2), dim=2)

        # 将句子向量维度变成 N*Ci*W*D
        # N为batch_size
        # Ci为通道数，文本为1
        # W为句子长度
        # D为词向量长度
        embedded_sentences = embedded_sentences.unsqueeze(1)

        # 分别用大小为2，3，5的卷积核卷积核进行卷积，并获得pooling后的值
        x2 = self.conv(embedded_sentences, self.conv2)
        x3 = self.conv(embedded_sentences, self.conv3)
        x5 = self.conv(embedded_sentences, self.conv5)

        # embedded_relation = embedded_entity1 - embedded_entity2  # N*1*D

        # 将x2,x3,x5，embedded_relation拼接起来，喂入全连接
        # out = self.out(torch.cat((x2, x3, x5, embedded_relation.squeeze(1)), dim=1))
        out = self.out(torch.cat((x2, x3, x5), dim=1))
        # out = self.out(torch.cat(( x3, embedded_relation), dim=1))
        return out


class CNNRE(nn.Module):
    '''
    CNN模型
    1. 将index转换为embedding
    2. 使用3*hidden_size的卷积核进行卷积
    3. 进行max pooling
    4. 与relation embedding 拼接后喂入全连接
    5. 返回out
    '''

    def __init__(self, hidden_size, out_size, voc):
        super(CNNRE, self).__init__()

        # self.embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        pe_size = 10
        self.word_embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        self.pos_embedding = torch.nn.Embedding(300 * 2, pe_size)

        Ci = 1
        Co = 8
        D = hidden_size + 2 * pe_size

        self.conv2 = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(2, D), padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(3, D), padding=(1, 0))
        self.conv5 = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(5, D), padding=(2, 0))

        self.out = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Co * 3 + hidden_size, out_size)
        )

    def conv(self, input, conv_func):
        x = conv_func(input)  # N*Co*W*1

        x = F.relu(x).squeeze(3)  # N*Co*W

        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # N*Co

        return x

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, position_to_entity1_batch,
                position_to_entity2_batch, hidden=None):
        embedded_sentences = self.word_embedding(sentences_seq.t())  # N*W*D
        embedded_entity1 = self.word_embedding(entity1_index)
        embedded_entity2 = self.word_embedding(entity2_index)
        pe1 = self.pos_embedding(position_to_entity1_batch.t())
        pe2 = self.pos_embedding(position_to_entity2_batch.t())
        embedded_sentences = torch.cat((embedded_sentences, pe1, pe2), dim=2)

        # 将句子向量维度变成 N*Ci*W*D
        # N为batch_size
        # Ci为通道数，文本为1
        # W为句子长度
        # D为词向量长度
        embedded_sentences = embedded_sentences.unsqueeze(1)

        # 分别用大小为2，3，5的卷积核卷积核进行卷积，并获得pooling后的值
        x2 = self.conv(embedded_sentences, self.conv2)
        x3 = self.conv(embedded_sentences, self.conv3)
        x5 = self.conv(embedded_sentences, self.conv5)

        embedded_relation = embedded_entity1 - embedded_entity2  # N*1*D

        # 将x2,x3,x5，embedded_relation拼接起来，喂入全连接
        out = self.out(torch.cat((x2, x3, x5, embedded_relation.squeeze(1)), dim=1))
        return out
