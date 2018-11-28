import myUtil


# 训练词向量
def train_w2v():
    myUtil.train_word2vec("../data/corpus", "../data/w2v.mdl", 64)

train_w2v();