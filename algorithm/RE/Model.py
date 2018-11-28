import torch
from torch import nn


class RE(nn.Module):
    '''
    Relation Embedding模型
    1. 将index转换为embedding
    2.
    3. 返回out
    '''

    def __init__(self, hidden_size, out_size, voc, dropout=0):
        super(RE, self).__init__()

        self.embedding = torch.nn.Embedding(voc.num_words, hidden_size)

        self.out = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, position_to_entity1_batch,
                position_to_entity2_batch, hidden=None):
        embedded_entity1 = self.embedding(entity1_index)
        embedded_entity2 = self.embedding(entity2_index)

        embedded_relation = embedded_entity1 - embedded_entity2

        out = self.out(embedded_relation.squeeze(1))
        return out
