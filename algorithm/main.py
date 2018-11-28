import numpy as np
import torch

import myUtil
from algorithm.Attn_GRU.Model import AttnGRU
from algorithm.LSTM.Model import LSTM
from algorithm.RE.Model import RE
from algorithm.CNN.Model import CNN
from algorithm.CNN.Model import CNNRE
from algorithm.Attn_CNN.Model import Attn_CNN
from algorithm.Transformer.Model import Transformer
from algorithm.Optimizer import NoamOpt
from myUtil import Voc
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

data_path = "../data/generatedBySystem_final.txt"


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
    positon_to_entity1_batch, positon_to_entity2_batch = myUtil.positionToEntity(sentence_indexes_batch,
                                                                                 entity1_indexes_batch,
                                                                                 entity2_indexes_batch)
    padList = torch.LongTensor(padList)
    entity1_indexes_batch = torch.LongTensor(entity1_indexes_batch)
    entity2_indexes_batch = torch.LongTensor(entity2_indexes_batch)
    tag_batch = torch.LongTensor(tag_batch)
    positon_to_entity1_batch = torch.LongTensor(myUtil.zeroPadding(positon_to_entity1_batch, fillvalue=0))
    positon_to_entity2_batch = torch.LongTensor(myUtil.zeroPadding(positon_to_entity2_batch, fillvalue=0))
    return padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch, positon_to_entity1_batch, positon_to_entity2_batch


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


def train(padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch, positon_to_entity1_batch,
          positon_to_entity2_batch, model,
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
    positon_to_entity1_batch = positon_to_entity1_batch.to(device)
    positon_to_entity2_batch = positon_to_entity2_batch.to(device)

    # 计算结果
    out = model(
        padList, lengths, entity1_indexes_batch, entity2_indexes_batch, positon_to_entity1_batch,
        positon_to_entity2_batch
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
    model.eval()
    # 用于测试，关闭梯度
    with torch.no_grad():
        total_correct = 0
        total_count = 0
        total_loss = 0
        for test_batch in testing_batches:
            padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch, \
            position_to_entity1_batch, position_to_entity2_batch = test_batch

            # 将相关变量放入 device中
            padList = padList.to(device)
            lengths = lengths.to(device)
            entity1_indexes_batch = entity1_indexes_batch.to(device)
            entity2_indexes_batch = entity2_indexes_batch.to(device)
            tag_batch = tag_batch.to(device)
            position_to_entity1_batch = position_to_entity1_batch.to(device)
            position_to_entity2_batch = position_to_entity2_batch.to(device)

            # 计算结果
            out = model(
                padList, lengths, entity1_indexes_batch, entity2_indexes_batch, position_to_entity1_batch,
                position_to_entity2_batch
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
    if batch_number * batch_size < len(trainPairs):
        training_batches.append(batchToTrainData(voc, trainPairs[batch_number * batch_size:len(trainPairs)]))

    # 生成testing_batches
    testing_batches = []
    for i in range(int(len(testPairs) / batch_size)):
        testing_batches.append(batchToTrainData(voc, testPairs[i * batch_size:i * batch_size + batch_size]))
    if int(len(testPairs) / batch_size) * batch_size < len(testPairs):
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
            padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch, positon_to_entity1_batch, positon_to_entity2_batch = train_batch

            loss, correct = train(padList, lengths, entity1_indexes_batch, entity2_indexes_batch, tag_batch,
                                  positon_to_entity1_batch, positon_to_entity2_batch, model,
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
        model = AttnGRU(hidden_size, out_size, voc, device, n_layers=1, dropout=0).to(device)

    if model_name == "LSTM":
        model = LSTM(hidden_size, out_size, voc, n_layers=2, dropout=0.5).to(device)

    if model_name == "RE":
        model = RE(hidden_size, out_size, voc, dropout=0.5).to(device)

    if model_name == "CNN":
        model = CNN(hidden_size, out_size, voc).to(device)

    if model_name == "CNNRE":
        model = CNNRE(hidden_size, out_size, voc).to(device)

    if model_name == "Attn_CNN":
        model = Attn_CNN(hidden_size, out_size, voc).to(device)

    if model_name == "Transformer":
        model = Transformer(hidden_size, out_size, voc).to(device)

    # 初始化网络参数
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model


def sample(pairs, ratio, random=False):
    '''
    进行同分布采样。按照每个类别所占的比率进行采样。
    :param pairs: 数据集总数
    :param ratio: 测试集总数
    :param random: 是否随机采样。默认为False，即采样最后几个。开启则随机采样。
    :return:
    '''
    map = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
    for pair in pairs:
        if pair[3] in map.keys():
            map[pair[3]].append(pair)

    trainPairs = []
    testPairs = []

    # 不是随机采样，
    if random == False:
        for key in map.keys():
            trainPairs += map[key][:int(len(map[key]) * (1 - ratio))]
            testPairs += map[key][int(len(map[key]) * (1 - ratio)):]
    else:
        for key in map.keys():
            sampleNumber = int(len(map[key]) * ratio)
            begin = random.randint(0, len(map[key]) - sampleNumber)
            trainPairs += map[key][:begin] + map[key][begin + sampleNumber:]
            testPairs += map[key][begin:begin + sampleNumber]

    return trainPairs, testPairs


# 主入口
def main(save_name, load_name=None, model_name="Attn_GRU"):
    print("加载数据......")
    pairs, voc = loadPrepareData()

    # 随机采样测试集
    trainPairs, testPairs = sample(pairs, 0.1)
    print("测试集数量{},训练集数量{}".format(len(trainPairs), len(testPairs)))

    print("初始化模型......")
    hidden_size = 64
    out_size = 9
    epoch = 150
    batch_size = 64
    lr = 0.001
    weight_decay = 3e-0

    # 设置当前的模型，加载参数
    model = get_model(model_name, hidden_size, out_size, voc)
    if load_name != None:
        model = torch.load('../data/model/{}.pkl'.format(load_name))
        print("模型加载成功!")

    # 设置optimizer与loss_func
    # optimizer = NoamOpt(hidden_size, 0.7, 800, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.999),
    #                                                             weight_decay=weight_decay))
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, weight_decay=weight_decay)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    # 开始训练
    try:
        train_losses, test_losses, train_accuracies, test_accuracies = \
            trainIters(epoch, batch_size, voc, trainPairs, testPairs, model, optimizer, loss_func)
        save_result(train_losses, test_losses, train_accuracies, test_accuracies, save_name, load_name)
        torch.save(model, '../data/model/{}.pkl'.format(save_name))
        print("模型保存成功!")
    finally:
        pass
        # torch.save(model, '../data/model/{}.pkl'.format(save_name))
        # print("模型保存成功!")


if __name__ == "__main__":
    # generatedBySystem_final
    # labeled_sentence
    # generatedBySystem_all
    data_path = "../data/generatedBySystem_final.txt"
    # Attn_GRU
    # LSTM
    # CNN
    # RE
    # Attn_CNN
    # Transformer
    # CNNRE
    main("Transformer_gsb6", load_name="Transformer_gsb5", model_name="Transformer")
