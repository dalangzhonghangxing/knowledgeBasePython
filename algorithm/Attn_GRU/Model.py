import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class EncoderRNN(nn.Module):
    '''
    1. 将word index转化为 embedding
    2. 计算relation embedding
    3. 计算每个word与relation embedding之间的attention权重
    4. 计算加权后的embedding，并进行pack padding.
    5. 计算bidirectional GRU结果的和
    6. 返回outputs与hidden
    '''

    def __init__(self, attn, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.attn = attn

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True, batch_first=True)

    def forward(self, sentences_seq, sentence_lengths, entity1_index, entity2_index, hidden=None):
        # 将句子与entity进行embedding
        embedded_sentences = self.embedding(sentences_seq)
        embedded_entity1 = self.embedding(entity1_index)
        embedded_entity2 = self.embedding(entity2_index)

        # 计算relation embedding
        embedded_relation = embedded_entity1-embedded_entity2

        attn_weights = self.attn(embedded_sentences,embedded_relation)

        weighted_sentence = attn_weights.bmm(embedded_sentences.transpose(0, 1))


        # 使用pack_padded_sequence对边长序列进行pack
        packed = torch.nn.utils.rnn.pack_padded_sequence(weighted_sentence, sentence_lengths, batch_first=True)

        outputs, hidden = self.gru(packed, hidden)

        # unpack outputs
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # 将双向的两个向量求和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(
            torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), dim=2)).tanch()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
