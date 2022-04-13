
import numpy as np
import math
from matplotlib.pylab import mpl
from MyUtils import *
import torch
from Model import CNN
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F #非线性函数在的地方


pos_files = ["u001_w001", "u001_w002", "u001_w003"]
neg_files = ["u002_w001", "u002_w002", "u002_w003"]
files = zip(pos_files, neg_files)


for index, (pos_file, neg_file) in enumerate(files):
    if index == 0:
        features, labels = CNN_row_dataTarget(pos_file, neg_file,sample_size=100)
        all_features = features
        all_labels = labels


    else:
        features, labels = CNN_row_dataTarget(pos_file, neg_file,sample_size=100)
        all_features = np.concatenate((all_features, features), axis=0)
        all_labels = np.concatenate((all_labels, labels), axis=0)

features = all_features
labels = all_labels
print(labels.shape)
features = features.astype(np.float32)
print(features.shape, labels.shape)


model = CNN()
batch_size = 15
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

features ,labels= torch.Tensor(features[:,:,0]),torch.Tensor(labels)
print(features.shape,labels.shape)
features, labels = features.reshape(batch_size,3,-1),labels.reshape(batch_size,3,-1) #N是一个批处理大小，C表示多个通道， L 是信号序列的长度
print(features.shape,labels.shape)
xTrain, xTest, yTrain, yTest = train_test_split(features,labels,test_size=0.3,random_state=222)



loss_count = []
for epoch in range():
    for i,(x,y) in enumerate(zip(xTrain,yTrain)):
        model.zero_grad()
        batch_x = x # torch.Size([128, 1, 28, 28])
        batch_y = y # torch.Size([128])
        print(batch_x.shape,batch_y.shape)
        # 获取最后输出
        out = model.forward(batch_x) # torch.Size([128,10])
        # 获取损失
        loss = loss_func(out,batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i%20 == 0:
            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())
            torch.save(model,r'C:\Users\liev\Desktop\myproject\yin_test\log_CNN')
        if i % 100 == 0:
            for a,b in (xTest,yTest):
                test_x = a
                test_y = b
                out = model(test_x)
                # print('test_out:\t',torch.max(out,1)[1])
                # print('test_y:\t',test_y)
                accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
                print('accuracy:\t',accuracy.mean())
                break

plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_count,label='Loss')
plt.legend()
plt.show()
