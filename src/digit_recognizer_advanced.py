import numpy as np  # scientific computing
import pandas as pd  # data analysis manipulation tool
import matplotlib.pyplot as plt  # static, animated, interactive visualization lib
# Scikit-learn is a free software machine learning library for the Python programming language

# train_test_split: a function in Sklearn classifier selection for splitting data arrays into two subsets: for training data
# and for testing data. With this function, you don't need to divide the dataset manually.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tqdm

"""
This advanced digit recognizer, implements KNN algorithms by itself, rather than import from lib
"""

print("Starting digit recognizer (advanced)...")

# <print data>
print("***********\n<print data>")
data_dir = '../data/'

pixel_width = pixel_height = 28


def load_data(data_dir, train_row):
    train_data = pd.read_csv(data_dir + 'train.csv', sep=',', header=0)
    print("\ntraining data set shape (rows-label, columns-feature): " + str(train_data.shape))
    X_train = train_data.values[:train_row, 1:]  # X_train: feature. column 0: label, [, 1:] exclude the 1st column
    y_train = train_data.values[:train_row, 0]  # y_train: label. column 0: label
    X_test = pd.read_csv(data_dir + 'test.csv', sep=',', header=0)
    return X_train, y_train, X_test


train_row = 5000
# Origin_X_train: feature of train data set
# Origin_y_train: label of test data set
# Origin_y_test: feature of test data set
Origin_X_train, Origin_y_train, Origin_y_test = load_data(data_dir, train_row)

print("\norigin x train shape: " + str(Origin_X_train.shape))
print("\norigin y train shape: " + str(Origin_y_train.shape))
print("\nX test data shape: " + str(Origin_y_test.shape))

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 4
print(classes)

# randomly print 4 rows of 9 digits
for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in Origin_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for i, idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_X_train[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)

plt.show()

# <train data>
print("\n\n***********\n<train data>")
# split the original training data set into 80% training data set, 20% validation set
X_train, X_valid, y_train, y_valid = train_test_split(Origin_X_train, Origin_y_train, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)


# implement KNN
class KNN:
    def __init__(self):
        pass

    def train(self, X, y):
        # for KNN, save the data, is the training!
        self.X_train = X
        self.y_train = y

    def predict(self, X, num, k=3):
        # find the Euclidean distance between the test data and the training data,
        # get the smallest Euclidean distance nodes
        dataset = self.X_train
        labels = self.y_train

        datasetSize = dataset.shape[0]
        # x (784,) dataset (4000, 784)

        diffMat = np.tile(X, (datasetSize, 1)) - dataset
        sqDiffMat = diffMat ** 2
        sumDiffMat = sqDiffMat.sum(axis=1)
        distances = sumDiffMat ** 0.5
        # argsort: return idx of elements after sorting in ascending order
        sortedDistances = distances.argsort()

        classCount = {}

        for i in range(k):
            vote = labels[sortedDistances[i]]
            classCount[vote] = classCount.get(vote, 0) + 1
        max, ans = 0, 0
        for k, v in classCount.items():
            if v > max:
                ans = k
                max = v

        # print("test #" + str(num + 1) + " prediction is " + str(ans))
        return ans


# KNN training
classifier = KNN()
# In KNN, train classifier is just save the training data set
classifier.train(X_train, y_train)

max_score = 0
ans_k = 0
for k in range(1, 7):
    print("\n--\nwhen k = " + str(k) + ", start training:")
    predictions = np.zeros(len(y_valid))
    for i in range(X_valid.shape[0]):
        if i % 500 == 0:
            print("Computing " + str(i + 1) + "/" + str(int(len(X_valid))) + "...")
        output = classifier.predict(X_valid[i], i, k)
        predictions[i] = output

    accuracy = accuracy_score(y_valid, predictions)
    print('k = ' + str(k), ' accuracy = ' + str(accuracy))
    if max_score < accuracy:
        ans_k = k
        max_score = accuracy

print('y_valid: ' + str(y_valid))
print('predictions: ' + str(predictions))


# test
print("\n\n***********\n<test data>")
k = ans_k
final_model = KNN()
final_model.train(Origin_X_train, Origin_y_train)

Origin_y_test = Origin_y_test[:300]
predictions = np.zeros(Origin_y_test.shape[0])
for i in range(Origin_y_test.shape[0]):  # show progress bar
    if i % 100 == 0:
        print("Computing " + str(i + 1) + "/" + str(int(len(Origin_y_test))) + "...")
    predictions[i] = classifier.predict(Origin_y_test.iloc[i, :], i, k)

idx = 298  # check a random predict
print(predictions[idx])
plt.imshow(Origin_y_test.iloc[idx].values.reshape(28, 28))
plt.show()


# save to file
print("\n\n***********\n<save data>")
df = pd.DataFrame({"ImageId": [i + 1 for i in range(len(predictions))], "Label": predictions})
df.to_csv("predictions.csv", index=False)
out_file = open('predictions.csv', 'w')
out_file.write('ImageId,Label\n')
for i in range(len(predictions)):
    out_file.write(str(i + 1) + ',' + str(int(predictions[i])) + '\n')
out_file.close()
