import tensorflow as tf 
from openpyxl import load_workbook
import numpy as np
import math


# Read the data from xlsx file
path = "/Users/apple/Documents/Github/enrollRec/stuData.xlsx"
workBook = load_workbook(path)                 # Load the file
dataSheet = workBook.get_sheet_by_name('TrainData')  # Get the datasheet

# From the file read the dataSet
row = 378742
col = 5
dataSet = np.zeros((row, col))
label = np.zeros((row, 1))

for i in range(0, row):
    for j in range(0, col):
        dataSet[i][j] = dataSheet.cell(row = i + 1, column = j + 1).value
    label[i] = dataSheet.cell(row = i + 1, column = 6).value
        
# Divide the dataSet into training set, test set and validation set
numTrainSet = math.ceil(row * 0.7)
numTestSet = math.ceil(row * 0.15)
numValSet = math.ceil(row * 0.15)
data_label_combined = np.hstack((dataSet, label))  # Combine the dataset and label to shuffle
np.random.shuffle(data_label_combined)             # Shuffle the dataset to divide
data_train_withlabel = data_label_combined[0:numTrainSet, :]
data_test_withlabel = data_label_combined[(numTrainSet - 1) : (numTrainSet + numTestSet), :]
data_val_withlabel = data_label_combined[(numTrainSet + numTestSet - 1) : row, :]




# Initialize the network




# Create the Model
X = tf.placeholder(tf.float32, [None, 5])

# Define the paras of the layers
# The weights
W1 = tf.Variable(tf.random_uniform([5, 12], -1.0, 1.0), name = "Weight1")
W2 = tf.Variable(tf.random_uniform([12, 12], -1.0, 1.0), name = "Weight2")
W3 = tf.Variable(tf.random_uniform([12, 12], -1.0, 1.0), name = "Weight3")
W4 = tf.Variable(tf.random_uniform([12, 10], -1.0, 1.0), name = "Weight4")

# Bias
b1 = tf.Variable(tf.zeros([12]), name = "Bias1")
b2 = tf.Variable(tf.zeros([12]), name = "Bias2")
b3 = tf.Variable(tf.zeros([12]), name = "Bias3")
b4 = tf.Variable(tf.zeros([10]), name = "Bias4")

# Hypothesis
with tf.name_scope("input") as scope:
    L1 = tf.sigmoid(tf.matul(X, W1) + b1)
with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matul(L1, W2) + b2)
with tf.name_scope("layer3") as scope:
    L3 = tf.sigmoid(tf.matul(L2, W3) + b3)
with tf.name_scope("output") as scope:
    hypothesis = tf.sigmoid(tf.matul(L3, W4) + b4) 
