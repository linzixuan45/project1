from MyUtils import *
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
    '''训练svm'''


    xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.3)

    SVM_best, score_train, score_test = get_best_svm_Auto(xTrain, xTest, yTrain, yTest, itera=100)
    print(xTrain.shape)
    _, ypre = SVM_best.predict(xTest)
    _, xpre = SVM_best.predict(xTrain)

    plt.plot(list(range(1, len(score_train) + 1)), score_train)
    plt.title("SVM Accuracy_score in Train", fontsize=20)
    plt.xlabel("The times of train")
    plt.ylabel("SVM Accuracy score")
    plt.savefig("SVM figures/train_score.jpg")
    plt.show()

    plt.plot(list(range(1, len(score_test) + 1)), score_test)
    plt.title("SVM Accuracy_score in Test", fontsize=20)
    plt.xlabel("The times of test")
    plt.ylabel("SVM Accuracy score")
    plt.savefig("SVM figures/test_score.jpg")
    plt.show()

    print(metrics.accuracy_score(ypre, yTest))
    print(metrics.accuracy_score(xpre, yTrain))
    print(yTest, yTrain)

    '''保存svm'''

    SVM_best.save(f"temp_file/cv2-svm-{features.shape[1]}.mat")
    '''保存特征'''



