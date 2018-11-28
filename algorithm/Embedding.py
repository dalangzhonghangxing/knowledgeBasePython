from torch import nn


class Embedding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, voc, dropout=0, max_len=300):
        super(Embedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #
        self.position_embedding = nn.Embedding(max_len * 2, d_model)
        self.word_embedding = nn.Embedding(voc.num_words, d_model)

    def forward(self, sentences_seq=None, position_to_entity1_batch=None, position_to_entity2_batch=None,
                entity_index=None):
        if entity_index is None:
            x = self.word_embedding(sentences_seq.t()) + self.position_embedding(
                position_to_entity1_batch.t()) + self.position_embedding(position_to_entity2_batch.t())
        else:
            x = self.word_embedding(entity_index)
        return self.dropout(x)
