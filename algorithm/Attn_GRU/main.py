import myUtil
import torch
from myUtil import Voc


def getPairs(lines):
    pairs = []
    for line in lines:
        values = line.split(" ")
        values[0] = myUtil.sentenceToWordList(values[0])
        pairs.append(values)
    return pairs


def loadPrepareData():
    print("Start preparing training data ...")
    lines = open("../../data/generatedBySystem.txt", encoding='utf-8'). \
        read().strip().split('\n')
    pairs = getPairs(lines)
    voc = Voc("train")
    print("Read {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
    print("Counted words:", voc.num_words)
    return pairs, voc


pairs, voc = loadPrepareData()

# 将一个batch的数据进行padding，并转换成tensor
def inputVar(sentence_batch, entity1_batch, entity2_batch, tag_batch, voc):
    sentence_indexes_batch = [myUtil.indexesFromSentence(voc, word_list) for word_list in sentence_batch]
    lengths = torch.tensor([len(indexes) for indexes in sentence_indexes_batch])
    padList = myUtil.zeroPadding(sentence_indexes_batch)
    entity1_indexes_batch = [voc.word2index(entity) for entity in entity1_batch]
    entity2_indexes_batch = [voc.word2index(entity) for entity in entity2_batch]
    padList = torch.LongTensor(padList)
    entity1_indexes_batch = torch.LongTensor(entity1_indexes_batch)
    entity2_indexes_batch = torch.LongTensor(entity2_indexes_batch)
    tag_batch = torch.LongTensor(tag_batch)
    return padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch


# 将一个batch转换成训练数据
def batchToTrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    sentence_batch = []
    entity1_batch = []
    entity2_batch = []
    tag_batch = []

    for pair in pair_batch:
        sentence_batch.append(pair[0])
        entity1_batch.append(pair[1])
        entity2_batch.append(pair[2])
        tag_batch.append([pair[3]])

    sentences, lengths = inputVar(input_batch, voc)
