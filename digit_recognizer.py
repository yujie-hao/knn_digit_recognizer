import numpy as np  # scientific computing
import pandas as pd  # data analysis manipulation tool
import matplotlib.pyplot as plt  # static, animated, interactive visualization lib
# Scikit-learn is a free software machine learning library for the Python programming language
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

print("Starting digit recognizer...")

# <print data>
print("***********\n<print data>")
data_dir = './data/'

pixel_width = pixel_height = 28


def load_data(data_dir, train_row):
    train_data = pd.read_csv(data_dir + 'train.csv', sep=',', header=0)
    print("\ntraining data set shape (rows-label, columns-feature): " + str(train_data.shape))
    x_train = train_data.values[:train_row, 1:]  # x_train: feature. column 0: label, [, 1:] exclude the 1st column
    y_train = train_data.values[:train_row, 0]  # y_train: label. column 0: label

    test_data = pd.read_csv(data_dir + 'test.csv', sep=',', header=0)
    return x_train, y_train, test_data


train_row = 5000
origin_x_train, origin_y_train, test_data = load_data(data_dir, train_row)

print("\norigin x train shape: " + str(origin_x_train.shape))
print("\norigin y train shape: " + str(origin_y_train.shape))
print("\ntest data shape: " + str(test_data.shape))

row = 3
print("\nlabel - row #" + str(row) + ": lable - " + str(origin_y_train[row]))
plt.imshow(origin_x_train[row].reshape((pixel_width, pixel_height)))  # reshape the 784 1D array to 28 x 28 2D array.
plt.show()  # plt.imshow - draw the picture; plt.show - show the picture

row = 16
print("\nlabel - row #" + str(row) + ": lable - " + str(origin_y_train[row]))
plt.imshow(origin_x_train[row].reshape((pixel_width, pixel_height)))
plt.show()

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 4

print(classes)
# randomly print 4 rows of 9 digits
for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in origin_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for i, idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(origin_x_train[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)

plt.show()

# <train data>

print("\n\n***********\n<train data>")
x_train, x_valid, y_train, y_valid = train_test_split(origin_x_train, origin_y_train, test_size=0.2, random_state=0)
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
k_range = range(1, 8)
scores = []

for k in k_range:
    print("=====\nk = " + str(k) + " --> train start:")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)  # store the value (training)
    y_pred = knn.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    scores.append(accuracy)

    end = time.time()
    print("time: " + str(end - start))
    print("accuracy: " + str(accuracy))
    print("\nclassification_report:\n" + str(classification_report(y_valid, y_pred)))
    print("\nconfusion_matrix:\n" + str(confusion_matrix(y_valid, y_pred)))

plt.plot(scores)
plt.show()  # use plot to get the best k --> k = 3 has the highest score (0.92)

# predict data
print("\n\n***********\npredict data\n")
res_k = 3  # get from the best `scores`
knn = KNeighborsClassifier(n_neighbors=res_k)
knn.fit(x_train, y_train)  # store the value (training)
y_pred = knn.predict(test_data[:100])

# show some test results:
row = 4
print(y_pred[row])
plt.imshow(test_data.values[row].reshape((28, 28)))
plt.show()

row = 10
print(y_pred[row])
plt.imshow(test_data.values[row].reshape((28, 28)))
plt.show()

row = 27
print(y_pred[row])
plt.imshow(test_data.values[row].reshape((28, 28)))
plt.show()
