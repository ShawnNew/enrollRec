#%% Recommendation system for students applying for university
# Author: Chenxiao Niu
# Date: Oct, 24,2017
# Version: 1.0.0
# Platform: TensorFlow
# @CopyRight

#%% Import Section
import tensorflow as tf 
from openpyxl import load_workbook
import numpy as np
import math

#%% Define next_batch function
def next_batch(num, data, labels):
    '''
    Return a total of 'num' random samples and labels.
    '''
    if len(data) == len(labels):
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    else:
        print 'The data set does not match with labels'
        


#%% Load the data from xlsx file
print('Loading the file...')
path = "/Users/apple/Documents/Github/enrollRec/stuData.xlsx"
workBook = load_workbook(path)                 # Load the file
dataSheet = workBook.get_sheet_by_name('TrainData')  # Get the datasheet
print 'Loading file Done...'

#%% From the file read the dataSet
print('Reading the file...')
row = 378742
col = 6
dataSet = np.zeros((row, col))


for i in range(0, row):
    for j in range(0, col):
        dataSet[i][j] = dataSheet.cell(row = i + 1, column = j + 1).value
    #label[i] = dataSheet.cell(row = i + 1, column = 6).value
print('Reading file done...')
   
     
#%% Divide the dataSet into training set, test set and validation set
print 'Construction the data...'
numTrainSet = int(math.ceil(row * 0.7))
numTestSet = int(math.ceil(row * 0.15)) - 1
numValSet = int(math.ceil(row * 0.15)) - 1
#data_label_combined = np.hstack((dataSet, label))  # Combine the dataset and label to shuffle

np.random.shuffle(dataSet)             # Shuffle the dataset to divide

# Divide the dataset
data_train_withlabel = dataSet[0: numTrainSet, :]
data_test_withlabel = dataSet[numTrainSet: (numTrainSet + numTestSet), :]
data_val_withlabel = dataSet[(numTrainSet + numTestSet) : (numTrainSet + numTestSet + numValSet), :]
# Map the label into one-hot vector
data_train = data_train_withlabel[:, 0 : 5]
data_val = data_val_withlabel[:, 0: 5]
data_test = data_test_withlabel[:, 0: 5]
# Vectorize the label
label_train = np.zeros((numTrainSet, 10))
label_test = np.zeros((numTestSet, 10))
label_val = np.zeros((numValSet, 10))
for i in range(len(data_train_withlabel)):
    for j in range(0, 10):
        label_train[i][(int(data_train_withlabel[i][5])) - 1] = 1.0
for i in range(len(data_test_withlabel)):
    for j in range(0, 10):
        label_test[i][(int(data_test_withlabel[i][5])) - 1] = 1.0
for i in range(len(data_val_withlabel)):
    for j in range(0, 10):
        label_val[i][(int(data_val_withlabel[i][5])) - 1] = 1.0

print 'Constructing the data done...'
print 'Size of Training Set:' + str(len(data_train))
print 'Size of Test Set:' + str(len(data_test))
print 'Size of Validation Set:' + str(len(data_val))

#%% Create the Graph
X = tf.placeholder(tf.float32, [None, 5])
Y = tf.placeholder(tf.float32, [None, 10])

# Define the paras of the layers
# The weights
keep_prob = tf.placeholder(tf.float32)
with tf.name_scope("Weight1") as scope:
    W1 = tf.Variable(tf.random_uniform([5, 12], -1.0, 1.0))
    W1_hist = tf.summary.histogram("Weight1", W1)
with tf.name_scope("Weight2") as scope:
    W2 = tf.Variable(tf.random_uniform([12, 12], -1.0, 1.0))
    W2_hist = tf.summary.histogram("Weight2", W2)
with tf.name_scope("Weight3") as scope:
    W3 = tf.Variable(tf.random_uniform([12, 12], -1.0, 1.0))
    W3_hist = tf.summary.histogram("Weight3", W3)
with tf.name_scope("Weight4") as scope:
    W4 = tf.Variable(tf.random_uniform([12, 10], -1.0, 1.0))
    W4_hist = tf.summary.histogram("Weight4", W4)

# Bias
b1 = tf.Variable(tf.zeros([12]), name = "Bias1")
b2 = tf.Variable(tf.zeros([12]), name = "Bias2")
b3 = tf.Variable(tf.zeros([12]), name = "Bias3")
b4 = tf.Variable(tf.zeros([10]), name = "Bias4")

# Hypothesis
with tf.name_scope("input") as scope:
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)  #dropout to prevent overfitting
with tf.name_scope("layer2") as scope:
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
with tf.name_scope("layer3") as scope:
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
with tf.name_scope("output") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L3, W4) + b4) 
    
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(hypothesis, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#%% Run the graph
# Initialize TensorFlow variablies
check_point = 200
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(10000):
# Iterate 10000 epoches, and every epoch with a batch of 100 data
    batch_xs, batch_ys = next_batch(100, data_train, label_train)
    sess.run(train, feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7})
    # hy = sess.run(hypothesis, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
    # print hy.shape
    if i%check_point ==0:
        #cost_at_training = sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        #print 'batch' + str(i) +' cost is: ' + str(cost_at_training)
        print(i, sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1}), sess.run([W1, W2, W3, W4]))


#%% Test model and check accuracy

# correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(label_train, 1))
# accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
acc = sess.run(accuracy, feed_dict={X: data_test, Y: label_test, keep_prob: 1})
print('Accuracy:' + str(acc))
    
    
    
    


