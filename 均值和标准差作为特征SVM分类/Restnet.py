import torchvision
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch import nn, optim
from torch.utils.data import DataLoader
from Model import *
import torch

# 准备的测试数据集
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("train数据集的长度为{}".format(train_data_size))
print("test数据集的长度为{}".format(test_data_size))

train = DataLoader(train_data, batch_size=64)
test = DataLoader(test_data, batch_size=64)

# 搭建神经网络 building your neural network
cnn = CNN()

# 损失函数loss
loss_function = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-3
print(learning_rate)
optimizer = optim.Adam(cnn.parameters(), learning_rate)

# 记录网络的一些参数

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("./model_training3")

# 训练
for i in range(epoch):
    print("--------第{}轮训练开始了--------".format(i + 1))

    cnn.train()
    for data in train:
        imgs, targets = data
        outputs = cnn(imgs)
        loss = loss_function(outputs, targets)
        # 梯度归零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 调用优化器
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{},Loss{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    cnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test:
            imgs, targets = data
            outputs = cnn(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss

            accuracy = (outputs.argmax(axis=1) == targets).sum()

            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss={}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_accuracy",total_accuracy / test_data_size,total_test_step)
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    total_test_step += 1

    torch.save(cnn, "cnn_{}.pth".format(i))
    print("模型已保存")

writer.close()