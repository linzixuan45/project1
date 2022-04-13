import numpy as np
import matplotlib.pyplot as plt
import sklearn
import csv
import pandas as pd
import datetime
import time
import skimage
from sklearn.metrics.pairwise import pairwise_distances #计算坐标点距离
import os
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def get_diff(ls):
  diff = []
  num = 0
  for index,data in enumerate(ls):
    if index+1<len(ls):
      m = [ls[index],ls[index + 1]]
      diff_1 = pairwise_distances(m)[0,1]
      print(diff_1)
      diff.append(diff_1)
      if diff_1 <0.1:
        num= num+1

  print(num)
  return diff

def stampToTime_16(stamp):
  datatime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(float(str(stamp)[0:9])))
  datatime = datatime+'.'+str(stamp)[9:]
  return datatime

def get_features(row_data, sample_size = 300):
    '''
    :param row_data: 原始的数据
    :return: （-1，3，2） 数据shape
    '''

    coordinate = row_data[:, 1:4]  # 三维坐标轴
    length = coordinate.shape[0]
    sample_length = int(length/sample_size)

    '''多的数据不要'''
    coordinate = coordinate[0:sample_length*sample_size]
    coordinate = coordinate.reshape(-1,sample_size,3)
    '''数据一阶段初步处理'''
    features = get_features_(coordinate)  # 每1份提取 6个特征,代表sample_size 长度的数据量  （-1，3，2）
    # print(f"This file:{file}   extract features shape is {coordinate.shape}")
    return features


def get_diff(ls):
  diff = []
  for index,data in enumerate(ls):
    if index+1<len(ls):
      diff.append(pairwise_distances(ls[index+1],ls[index],metric="euclidean"))
  return diff


def get_features_(data):
    '''

    :param data: 时间序列数据类型
    :return: (-1, 3, 2)
    '''
    feature = []
    shape = data.shape
    for shape0 in data:
        feature.append(np.mean(shape0[:, 0]))
        feature.append(np.var(shape0[:, 0]))
        feature.append(np.mean(shape0[:, 1]))
        feature.append(np.var(shape0[:, 1]))
        feature.append(np.mean(shape0[:, 2]))
        feature.append(np.var(shape0[:, 2]))

    feature = np.array(feature).reshape(-1, 3, 2)
    return feature

def seconde(data):
    Time = data[:,0]
    Time = list(map(lambda x : int(str(x)[0:9]), Time))
    pass


def hysteresis_(data,hysteresis_num):
    '''

    :param data: (-1, 3, 2)类型
    :param hysteresis_num:
    :return: data.shape[0] - hysteresis_num 个数据,shape：(-1, 3, 2, hysteresis_num) 3为坐标维度   2为 均值和方差

    '''
    num = hysteresis_num
    expansion_data = []
    for index,shape1 in enumerate(data):
        if index + num <= data.shape[0]:
            b = data[list(map(lambda x: x, range(index, index + num)))]
            new_b = np.concatenate((b[:, :, 0], b[:, :, 1]), axis=0).reshape(3, 2, num)
            expansion_data.append(new_b)
    expansion_data = np.array(expansion_data)
    return expansion_data

def CNN_row_dataTarget(pos_file,neg_file,sample_size = 100):
    '''
    :param pos_file: positive file root
    :param neg_file: negative file root
    :return: 返回TCN的数据和标签 shape : (  number, 3,2) , label
    '''

    all_features, all_labels = [], []
    pos_features, pos_labels = [], 0
    neg_features, neg_labels = [], 0

    '''读取文件夹下的所以文件'''
    for index, file in enumerate(os.listdir(pos_file)):
        file = pos_file + r'/{}'.format(file)
        data = pd.read_csv(file,delimiter="\t")
        data = np.array(data)

        data = get_features(data,sample_size) # (-1,3,2)
        print(data.shape)
        pos_features.append(data)  # (-1,3,2,hysteresis_num)

    for index,file_value in enumerate(pos_features):
        if index == 0:
            all_features = np.array(file_value)
            pos_labels = file_value.shape[0]
        else:
            all_features = np.concatenate((all_features,file_value),axis=0)
            pos_labels = pos_labels+file_value.shape[0]

    pos_labels = np.ones(pos_labels)
    print(f'pos_features {all_features.shape}, pos_labels is {pos_labels.shape}')

    for index, file in enumerate(os.listdir(neg_file)):
        file = neg_file + r'/{}'.format(file)
        data = pd.read_csv(file,delimiter="\t")
        data =np.array(data)

        data = get_features(data,sample_size)
        neg_features.append(data)  # 1000x6

    for index,file_value in enumerate(neg_features):
        if index == 0:
            all_features = np.concatenate((all_features, file_value), axis=0)
            neg_labels = file_value.shape[0]
        else:
            all_features = np.concatenate((all_features,file_value),axis=0)
            neg_labels = neg_labels+file_value.shape[0]

    neg_labels = np.zeros(neg_labels)
    print(f'all_features shape is {all_features.shape}, neg_labels is {neg_labels.shape}')

    '''连接所有标签'''
    all_labels = np.concatenate((pos_labels,neg_labels))
    print(f"特征维度为：{all_features.shape},标签维度为：{all_labels.shape}")
    return all_features,all_labels


def TCN_row_dataTarget(pos_file,neg_file,hysteresis_num):

    '''
    :param pos_file: positive file root
    :param neg_file: negative file root
    :param hysteresis_num: 数据的TCN滞后数据，滞后扩充数据
    :return: 返回TCN的数据和标签 shape : (  number, 3,2, hysteresis_num),label,并保存在temp_file 文件夹下
    '''

    all_features, all_labels = [], []
    pos_features, pos_labels = [], 0
    neg_features, neg_labels = [], 0

    '''读取文件夹下的所以文件'''
    for index, file in enumerate(os.listdir(pos_file)):
        file = pos_file + r'/{}'.format(file)
        data = pd.read_csv(file,delimiter="\t")
        data = np.array(data)


        data = get_features(data) # (-1,3,2)
        print(data.shape)
        pos_features.append(hysteresis_(data,hysteresis_num))  # (-1,3,2,hysteresis_num)

    for index,file_value in enumerate(pos_features):
        if index == 0:
            all_features = np.array(file_value)
            pos_labels = file_value.shape[0]
        else:
            all_features = np.concatenate((all_features,file_value),axis=0)
            pos_labels = pos_labels+file_value.shape[0]

    pos_labels = np.ones(pos_labels)
    print(f'pos_features {all_features.shape}, pos_labels is {pos_labels.shape}')

    for index, file in enumerate(os.listdir(neg_file)):
        file = neg_file + r'/{}'.format(file)
        data = pd.read_csv(file,delimiter="\t")
        data =np.array(data)

        data = get_features(data)
        neg_features.append(hysteresis_(data,hysteresis_num))  # 1000x6

    for index,file_value in enumerate(neg_features):
        if index == 0:
            all_features = np.concatenate((all_features, file_value), axis=0)
            neg_labels = file_value.shape[0]
        else:
            all_features = np.concatenate((all_features,file_value),axis=0)
            neg_labels = neg_labels+file_value.shape[0]

    neg_labels = np.zeros(neg_labels)
    print(f'all_features shape is {all_features.shape}, neg_labels is {neg_labels.shape}')

    '''连接所有标签'''
    all_labels = np.concatenate((pos_labels,neg_labels))
    print(f"特征维度为：{all_features.shape},标签维度为：{all_labels.shape}")

    return all_features,all_labels


def get_sampleFeatures_label(pos_file, neg_file):  # 传入正的文件夹，和负的文件夹
    '''在用这个，传入pos 文件夹，neg文件夹自动返回对应的特征'''
    all_features, all_labels = [], []
    pos_features ,pos_labels= [],0
    neg_features,neg_labels = [],0


    '''读取文件夹下的所以文件'''
    for index, file in enumerate(os.listdir(pos_file)):
        file = pos_file + r'/{}'.format(file)
        data = pd.read_csv(file,delimiter="\t")
        data =np.array(data)
        Time = data[:,0]

        pos_features.append(get_features(data,file)) # 1000x6


    for index,file_value in enumerate(pos_features):
        if index == 0:
            all_features = np.array(file_value)
            pos_labels = file_value.shape[0]
        else:
            all_features = np.concatenate((all_features,file_value),axis=0)
            pos_labels = pos_labels+file_value.shape[0]

    pos_labels = np.ones(pos_labels)
    print(f'pos_features {all_features.shape}, pos_labels is {pos_labels.shape}')


    for index, file in enumerate(os.listdir(neg_file)):
        file = neg_file + r'/{}'.format(file)
        data = pd.read_csv(file,delimiter="\t")
        data =np.array(data)
        Time = data[:,0]

        neg_features.append(get_features(data,file)) # 1000x6


    for index,file_value in enumerate(neg_features):
        if index == 0:
            all_features = np.concatenate((all_features, file_value), axis=0)
            neg_labels = file_value.shape[0]
        else:
            all_features = np.concatenate((all_features,file_value),axis=0)
            neg_labels = neg_labels+file_value.shape[0]

    neg_labels = np.zeros(neg_labels)
    print(f'all_features shape is {all_features.shape}, neg_labels is {neg_labels.shape}')



    '''连接所有标签'''
    all_labels = np.concatenate((pos_labels,neg_labels))

    print(f"特征维度为：{all_features.shape},标签维度为：{all_labels.shape}")

    return all_features,all_labels

def creat_svm_Auto(training_attributes,training_class_labels):
    '''这个默认rbf，且每次都进行1000次的参数选择，交叉验证10个子集'''
    svm = cv2.ml.SVM_create()
    svm.trainAuto(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int),
                  kFold=10);# 2分类
    return svm

def score_svm(svm, xTest, yTest):
    """svm, xTest, yTest"""
    from sklearn.metrics import accuracy_score
    _, y_pre = svm.predict(xTest)
    return accuracy_score(y_pre, yTest)


def get_best_svm_Auto(xTrain, xTest, yTrain, yTest,itera = 100):
    '''itera,迭代次数'''
    score_train, score_test = [], []
    for j in range(itera):

        svm = creat_svm_Auto(xTrain, yTrain)
        score_train.append(score_svm(svm, xTrain, yTrain))
        score_test.append(score_svm(svm, xTest, yTest))
        _, y_pred = svm.predict(xTest)
        '''将预测失败的图片添加进train中'''
        flag1, flag2 = yTest.ravel() == 0, y_pred.ravel() == 1
        flag = flag1 == flag2  # flag 中记载了所有预测失败的位置信息
        xTrain_append = xTest[flag, :]  # (40, 18000, 1)
        yTrain_append = yTest[flag]  # 1维数据

        print(f"目前为第{j+1}次训练，训练精度为：{score_train[j]},测试集准确度为：{score_test[j]}")
        if yTrain_append.shape[0] == 0:  # 已经没有预测错误的数据了
            print('no more false predict')
            return svm, score_train, score_test

        else:
            xTrain = np.concatenate((xTrain, xTrain_append), axis=0)
            yTrain = np.concatenate((yTrain, yTrain_append))

    return svm, score_train, score_test

