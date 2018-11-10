import numpy as np
import torch

import myUtil
from algorithm.Attn_GRU.Model import Attn
from algorithm.Attn_GRU.Model import AttnGRU
from algorithm.LSTM.Model import LSTM
from algorithm.RE.Model import RE
from algorithm.CNN.Model import CNN
from myUtil import Voc
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

data_path = "../data/generatedBySystem.txt"


def getPairs(lines):
    pairs = []
    for line in lines:
        values = line.split(" ")
        values[0] = myUtil.sentenceToWordList(values[0])
        pairs.append(values)
    return pairs


def loadPrepareData():
    print("Start preparing training data ...")
    lines = open(data_path, encoding='utf-8'). \
        read().strip().split('\n')
    pairs = getPairs(lines)
    voc = Voc("train")
    print("Read {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
    print("Counted words:", voc.num_words)
    return pairs, voc


# 将一个batch的数据进行padding，并转换成tensor
def inputVar(sentence_batch, entity1_batch, entity2_batch, tag_batch, voc):
    sentence_indexes_batch = [myUtil.indexesFromSentence(voc, word_list) for word_list in sentence_batch]
    lengths = torch.tensor([len(indexes) for indexes in sentence_indexes_batch])
    padList = myUtil.zeroPadding(sentence_indexes_batch)

    entity1_indexes_batch = [voc.word2index[entity] for entity in entity1_batch]
    entity2_indexes_batch = [voc.word2index[entity] for entity in entity2_batch]
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
        try:
            if int(pair[3]) <= 0 or int(pair[3]) > 9: continue
            sentence_batch.append(pair[0])
            entity1_batch.append(pair[1])
            entity2_batch.append(pair[2])
            tag_batch.append(int(pair[3]) - 1)
        except:
            print(pair[0])

    return inputVar(sentence_batch, entity1_batch, entity2_batch, tag_batch, voc)


def train(padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch, model,
          optimizer, loss_func):
    '''
    训练一个batch
    :param padList: padding的句子
    :param lengths: 每句句子的长度
    :param entity1_indexes_batch:entity1 列表
    :param entity2_indexes_batch: entity2 列表
    :param tag_batch: 关系标签
    :param model: 模型
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :return:
    '''
    optimizer.zero_grad()

    # 将相关变量放入 device中
    padList = padList.to(device)
    lengths = lengths.to(device)
    entity1_indexes_batch = entity1_indexes_batch.to(device)
    entity2_indexes_batch = entity2_indexes_batch.to(device)
    tag_batch = tag_batch.to(device)

    # 计算结果
    out = model(
        padList, lengths, entity1_indexes_batch, entity2_indexes_batch
    )

    # 计算loss
    loss = loss_func(out, tag_batch)

    # 计算准确个数
    pre = torch.max(out, dim=1)[1]
    correct = torch.sum(pre == tag_batch)

    # 反向传播更新梯度
    loss.backward()
    optimizer.step()

    return loss, correct


def test(testing_batches, model, loss_func):
    # 用于测试，关闭梯度
    with torch.no_grad():
        total_correct = 0
        total_count = 0
        total_loss = 0
        for test_batch in testing_batches:
            padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch = test_batch

            # 将相关变量放入 device中
            padList = padList.to(device)
            lengths = lengths.to(device)
            entity1_indexes_batch = entity1_indexes_batch.to(device)
            entity2_indexes_batch = entity2_indexes_batch.to(device)
            tag_batch = tag_batch.to(device)

            # 计算结果
            model.eval()
            out = model(
                padList, lengths, entity1_indexes_batch, entity2_indexes_batch
            )

            # 计算loss
            loss = loss_func(out, tag_batch)

            # 计算准确个数
            pre = torch.max(out, dim=1)[1]
            correct = torch.sum(pre == tag_batch)

            # 计算测试集上面的 loss、correct
            total_loss += loss
            total_correct += correct
            total_count += entity1_indexes_batch.shape[0]
        print("测试集       :总loss: {:.5f} ;正确率: {:.5f}\n".format(total_loss.cpu().data.numpy().item(),
                                                               total_correct.cpu().data.numpy().item() / total_count))
        return total_loss.cpu().data.numpy().item(), total_correct.cpu().data.numpy().item() / total_count


def draw(train, test, title):
    # 设置x轴
    x = range(len(train))
    steps = [i for i in x]
    plt.figure()
    plt.plot(x, test, marker='o', mfc='w', label=u'test_' + title)
    plt.plot(x, train, marker='o', label=u'train_' + title)
    plt.legend()  # 让图例x生效
    plt.xticks(x, steps, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"epoch")  # X轴标签
    plt.ylabel("accuracy")  # Y轴标签
    plt.title(title)  # 标题
    plt.show()


def trainIters(epoch, batch_size, voc, trainPairs, testPairs, model, optimizer, loss_func):
    batch_number = int(len(trainPairs) / batch_size)
    train_number = len(trainPairs)
    test_number = len(testPairs)
    # 随机打乱trainPairs
    np.random.shuffle(trainPairs)

    # 生成training_batches
    training_batches = []
    for i in range(batch_number):
        training_batches.append(batchToTrainData(voc, trainPairs[i * batch_size:i * batch_size + batch_size]))
    training_batches.append(batchToTrainData(voc, trainPairs[batch_number * batch_size:len(trainPairs)]))

    # 生成testing_batches
    testing_batches = []
    for i in range(int(len(testPairs) / batch_size)):
        testing_batches.append(batchToTrainData(voc, testPairs[i * batch_size:i * batch_size + batch_size]))
    testing_batches.append(
        batchToTrainData(voc, testPairs[int(len(testPairs) / batch_size) * batch_size:len(testPairs)]))

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for e in range(epoch):
        # 初始化
        train_loss = 0
        correct_total = 0
        total_count = 0
        model.train()

        # 依次训练所有batch
        for train_batch in training_batches:
            padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch = train_batch

            loss, correct = train(padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch, model,
                                  optimizer, loss_func)

            train_loss += loss
            correct_total += correct
            total_count += entity1_indexes_batch.shape[0]

        # 输出一个batch的结果
        print("当前epoch: {} ;总loss: {:.5f} ;正确率: {:.5f}".format(e, train_loss.cpu().data.numpy().item(),
                                                               correct_total.cpu().data.numpy().item() / total_count))
        test_loss, test_accuracy = test(testing_batches, model, loss_func)

        train_losses.append(train_loss.cpu().data.numpy().item() / train_number)
        train_accuracies.append(correct_total.cpu().data.numpy().item() / total_count)
        test_losses.append(test_loss / test_number)
        test_accuracies.append(test_accuracy)

    # draw(train_losses, test_losses, "loss")
    # draw(train_accuracies, test_accuracies, "accuracy")
    return train_losses, test_losses, train_accuracies, test_accuracies


# 保存中间结果，用来画图
def save_result(train_losses, test_losses, train_accuracies, test_accuracies, save_name, load_name):
    content = ""
    for tl in train_losses:
        content += str(tl) + " "
    content += "\n"

    for tl in test_losses:
        content += str(tl) + " "
    content += "\n"

    for ta in train_accuracies:
        content += str(ta) + " "
    content += "\n"

    for ta in test_accuracies:
        content += str(ta) + " "
    content += "\n"

    # 更加model_name是否为None来判断是否追加结果
    if load_name is None:
        myUtil.writeFile("../data/result/" + save_name + ".result", content)
    else:
        myUtil.writeFile_Add("../data/result/" + save_name + ".result", content)
    print("中间结果已保存!")


# 根据模型名称，获取模型
def get_model(model_name, hidden_size, out_size, voc):
    if model_name == "Attn_GRU":
        # concat general dot
        attn = Attn("general", hidden_size).to(device)
        model = AttnGRU(attn, hidden_size, out_size, voc, n_layers=1, dropout=0.5).to(device)

    if model_name == "LSTM":
        model = LSTM(hidden_size, out_size, voc, n_layers=2, dropout=0.5).to(device)

    if model_name == "RE":
        model = RE(hidden_size, out_size, voc, dropout=0.5).to(device)

    if model_name == "CNN":
        model = CNN(hidden_size, out_size, voc, dropout=0.5).to(device)
    return model


# 主入口
def main(save_name, load_name=None, model_name="Attn_GRU"):
    print("加载数据......")
    pairs, voc = loadPrepareData()

    # 随机采样测试集
    sampleNumber = 400
    testBegin = random.randint(0, len(pairs) - sampleNumber)
    trainPairs = pairs[:testBegin] + pairs[testBegin + sampleNumber:]
    testPairs = pairs[testBegin:testBegin + sampleNumber]

    print("测试集数量{},训练集数量{}".format(len(trainPairs), len(testPairs)))

    print("初始化模型......")
    hidden_size = 64
    out_size = 9
    epoch = 1000
    batch_size = 64
    lr = 0.01

    # 设置当前的模型，加载参数
    model = get_model(model_name, hidden_size, out_size, voc)
    if load_name != None:
        model = torch.load('../data/model/{}.pkl'.format(load_name))
        print("模型加载成功!")

    # 设置optimizer与loss_func
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adagrad(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    # 开始训练
    try:
        train_losses, test_losses, train_accuracies, test_accuracies = \
            trainIters(epoch, batch_size, voc, trainPairs, testPairs, model, optimizer, loss_func)
        save_result(train_losses, test_losses, train_accuracies, test_accuracies, save_name, load_name)
    finally:
        torch.save(model, '../data/model/{}.pkl'.format(save_name))
        print("模型保存成功!")


if __name__ == "__main__":
    # generatedBySystem
    data_path = "../data/generatedBySystem.txt"
    # Attn_GRU  LSTM CNN RE
    # Attn_GRU_hid32
    main("RE_gsb", model_name="RE")
