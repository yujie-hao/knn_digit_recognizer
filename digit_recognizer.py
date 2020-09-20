import numpy as np  # scientific computing
import pandas as pd  # data analysis manipulation tool
import matplotlib.pyplot as plt  # static, animated, interactive visualization lib
import time

print("Starting digit recognizer...")

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

