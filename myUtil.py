import unicodedata
import re
import itertools
import jieba
from gensim.models import Word2Vec
import os

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
jieba.load_userdict("../wordBase.txt")


# 用于记录单词表
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, wordList):
        for word in wordList:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


# 将文件按行读入到list中，为了训练word2vec
class Sentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = normalizeString(line)
                if line != "":
                    yield list(jieba.cut(line))


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 标准化处理
def normalizeString(s):
    s = s.strip()
    if "《" in s or "年" in s or "∵" in s or "∴" in s or len(s) < 6:
        return ""
    s.replace("（", "(")
    s.replace("）", ")")
    s.replace("，", ",")
    s.replace(".", "。")
    s.replace("；", ";")
    s.replace("：", ":")
    s.replace("①", "1、")
    s.replace("②", "2、")
    s.replace("③", "3、")
    s.replace("④", "4、")
    s.replace("°", "度")
    s.replace("．", "、")
    s.replace("√", "根号")
    s.replace("²", "^2")
    return s


def indexesFromSentence(voc, word_list):
    return [voc.word2index[word] for word in word_list] + [EOS_token]


def positionsFromSentence(sentence, word_index):
    ans = []
    for i, index in enumerate(sentence):
        if word_index == index:
            ans.append(i)
    return ans


def positionToEntity(sentence_indexes_batch, entity1_indexes_batch, entity2_indexes_batch):
    positon_to_entity1_batch = []
    positon_to_entity2_batch = []
    for index in range(len(sentence_indexes_batch)):
        sentence = sentence_indexes_batch[index]
        entity1 = entity1_indexes_batch[index]
        entity2 = entity2_indexes_batch[index]

        entity1_potisions = positionsFromSentence(sentence, entity1)
        entity2_potisions = positionsFromSentence(sentence, entity2)

        positon_to_entity1 = []
        positon_to_entity2 = []
        for i in range(len(sentence)):
            positon_to_entity1.append(closePosition(entity1_potisions, i))
            positon_to_entity2.append(closePosition(entity2_potisions, i))
        positon_to_entity1_batch.append(positon_to_entity1)
        positon_to_entity2_batch.append(positon_to_entity2)
    return positon_to_entity1_batch, positon_to_entity2_batch


def closePosition(entity_positions, index):
    '''
    获取到最近的指定entity的距离，距离为有向距离。为了保证relative_position为正，最后加上一个常数来将负距离变为正数
    :param entity_potisions: 指定entity在sentence中的position list
    :param index: 当前单词在sentence中的index
    :return:
    '''
    _abs = 100000
    relative_position = 0
    for position in entity_positions:
        if abs(index - position) < _abs:
            _abs = abs(index - position)
            relative_position = index - position
    return relative_position + 300


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def sentenceToWordList(sentence):
    return list(jieba.cut(sentence))


def writeFile(filepath, content):
    output = open(filepath, 'w', encoding='UTF-8')
    output.write(content)
    output.close()


def writeFile_Add(filepath, content):
    output = open(filepath, 'a', encoding='utf-8')
    output.write(content)
    output.close()


# 根据语料所在文件夹，训练词向量
def train_word2vec(folder_path, save_path, size=100):
    sentences = Sentences(folder_path)
    model = Word2Vec(sentences, size=size, workers=8, min_count=0)
    model.save(save_path)
