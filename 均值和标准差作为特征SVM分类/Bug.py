import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.layers import LeakyReLU
from tcn import TCN,tcn_full_summary
from sklearn.metrics import mean_squared_error # 均方误差
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model,Sequential
from MyUtils import *



pos_files = ["u001_w001", "u001_w002", "u001_w003"]
neg_files = ["u002_w001", "u002_w002", "u002_w003"]
files = zip(pos_files, neg_files)


for index, (pos_file, neg_file) in enumerate(files):
    if index == 0:
        features, labels = TCN_row_dataTarget(pos_file, neg_file,hysteresis_num=20)
        all_features = features
        all_labels = labels


    else:
        features, labels = TCN_row_dataTarget(pos_file, neg_file,hysteresis_num=20)
        all_features = np.concatenate((all_features, features), axis=0)
        all_labels = np.concatenate((all_labels, labels), axis=0)

features = all_features
labels = all_labels
print(labels.shape)
features = features.astype(np.float32)
print(features.shape, labels.shape)
