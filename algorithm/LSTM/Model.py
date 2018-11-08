import torch
from torch import nn


class LSTM(nn.Module):
    '''
    双向LSTM模型
    1. 将index转换为embedding
    2. 将embedded_sentence喂入lstm
    3. 将双向lstm的结果求和
    4. 将lstm的最后一层输出与embedded_relation进行concat，喂入全连接
    5. 返回out
    '''

    def __init__(self, hidden_size, out_size, voc, n_layers=1, dropout=0):
        super(LSTM, self).__init__()

        self.embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        self.num_layer = n_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.out = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, out_size)
        )

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, hidden=None):
        # 将index转换为embedding
        # sentences_seq是length*batch的，所以要先进性转置
        embedded_sentence = self.embedding(sentences_seq.t())
        embedded_entity1 = self.embedding(entity1_index)
        embedded_entity2 = self.embedding(entity2_index)

        embedded_relation = embedded_entity1 - embedded_entity2

        # pack
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded_sentence, sentence_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(packed, hidden)

        # unpack
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # 将双向向量求和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        out = self.out(torch.cat((outputs[:, -1, :], embedded_relation.squeeze(1)), dim=1))

        return out
