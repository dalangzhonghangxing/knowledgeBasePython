import torch
import torch.nn.functional as F
import math


class Attn(torch.nn.Module):
    '''
    Attn模型
    基于论文Effective Approaches to Attention-based Neural Machine Translation实现
    scaled为 Attention is All your need中的实现
    '''

    def __init__(self, method, query_size, key_size, dropout=0.5):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat', 'scaled']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        if query_size is None:  # 使用特定的Attention，不初始化下面的参数
            pass

        self.query_size = query_size

        if self.method == 'general':
            self.attn = torch.nn.Linear(key_size, query_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(key_size+query_size, key_size, dropout=dropout)
            self.v = torch.nn.Parameter(torch.FloatTensor(key_size))

    def dot_score(self, query, key):
        return torch.sum(query * key, dim=2)

    def general_score(self, query, key):
        energy = self.attn(key)
        return torch.sum(query * energy, dim=2)

    def concat_score(self, query, key):
        energy = self.attn(
            torch.cat((query.expand(key.size(0), -1, -1), key), dim=2)).tanch()
        return torch.sum(self.v * energy, dim=2)

    def scaled_score(self, query, key, value, mask, dropout):
        if value is None:
            raise ValueError("Attention error:value can not be None!")
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value=None, mask=None, dropout=None):
        '''
        根据target，计算weighted的权重
        :param query: 目标
        :param key: 待计算权重向量
        :param value: 需要加权的value
        :return: 权重
        '''
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'scaled':
            return self.scaled_score(query, key, value, mask, dropout)
        if self.method == 'general':
            attn_energies = self.general_score(query, key)
        elif self.method == 'concat':
            attn_energies = self.concat_score(query, key)
        elif self.method == 'dot':
            attn_energies = self.dot_score(query, key)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
