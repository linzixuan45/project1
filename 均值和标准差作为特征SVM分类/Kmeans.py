from MyUtils import  *
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    pos_files = ["u001_w001", "u001_w002", "u001_w003"]
    neg_files = ["u002_w001", "u002_w002", "u002_w003"]
    files = zip(pos_files, neg_files)

    for index, (pos_file, neg_file) in enumerate(files):
        if index == 0:
            features, labels = get_sampleFeatures_label(pos_file, neg_file)
            all_features = features
            all_labels = labels


        else:
            features, labels = get_sampleFeatures_label(pos_file, neg_file)
            all_features = np.concatenate((all_features, features), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)

    features = all_features
    labels = all_labels
    print(labels.shape)
    features = features.astype(np.float32)
    print(features.shape, labels.shape)

    # label int ,feature 3维度

    xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.3)

    """                           Kmeans 聚类                                  """
    '''-----------------------------------------------------------------------'''

    k_means = KMeans(n_clusters=2)
    k_means.fit(features)
    y_preTrain = k_means.predict(xTrain)
    y_preTest = k_means.predict(xTest)
    print(y_preTrain.shape, y_preTest.shape)  # (1402,) (601,)
    print(f"训练集的准确率：{metrics.accuracy_score(y_preTrain, yTrain)}, 测试集的准确率：{metrics.accuracy_score(y_preTest, yTest)}")
    print(f"训练集的召回率：{metrics.recall_score(y_preTrain, yTrain)}, 测试集的召回率：{metrics.accuracy_score(y_preTest, yTest)}")
