import tensorflow as tf 
from openpyxl import load_workbook
import numpy as np
import math

# Define next_batch function
def next_batch(num, data, labels):
    '''
    Return a total of 'num' random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# Read the data from xlsx file
path = "/Users/apple/Documents/Github/enrollRec/stuData.xlsx"
workBook = load_workbook(path)                 # Load the file
dataSheet = workBook.get_sheet_by_name('TrainData')  # Get the datasheet

# From the file read the dataSet
row = 378742
col = 6
dataSet = np.zeros((row, col))
label_train = np.zeros((row, 10))
label_test = np.zeros((row, 10))
label_val = np.zeros((row, 10))

for i in range(0, row):
    for j in range(0, col):
        dataSet[i][j] = dataSheet.cell(row = i + 1, column = j + 1).value
    #label[i] = dataSheet.cell(row = i + 1, column = 6).value
   
     
# Divide the dataSet into training set, test set and validation set
numTrainSet = math.ceil(row * 0.7)
numTestSet = math.ceil(row * 0.15)
numValSet = math.ceil(row * 0.15)
#data_label_combined = np.hstack((dataSet, label))  # Combine the dataset and label to shuffle

#np.random.shuffle(dataSet)             # Shuffle the dataset to divide

# Divide the dataset
data_train_withlabel = dataSet[0:numTrainSet, :]
data_test_withlabel = dataSet[(numTrainSet - 1) : (numTrainSet + numTestSet), :]
data_val_withlabel = dataSet[(numTrainSet + numTestSet - 1) : row, :]
# Map the label into one-hot vector
data_train = data_train_withlabel[:, 0 : 5]
data_val = data_val_withlabel[:, 0: 5]
data_test = data_test_withlabel[:, 0: 5]
# Vectorize the label
for i in range(0, row):
    for j in range(0, 10):
        label_train[i][(data_train_withlabel[i][6]) - 1] = 1
        label_test[i][(data_test_withlabel[i][6]) - 1] = 1
        label_val[i][(data_val_withlabel[i][6]) - 1] = 1



# Create the Graph
X = tf.placeholder(tf.float32, [None, 5])
Y = tf.placeholder(tf.float32, [None, 10])

# Define the paras of the layers
# The weights
keep_prob = tf.placeholder(tf.float32)
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
    L1 = tf.nn.relu(tf.matul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)  #dropout to prevent overfitting
with tf.name_scope("layer2") as scope:
    L2 = tf.nn.relu(tf.matul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
with tf.name_scope("layer3") as scope:
    L3 = tf.nn.relu(tf.matul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
with tf.name_scope("output") as scope:
    hypothesis = tf.nn.relu(tf.matul(L3, W4) + b4) 
    
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)



# Run the graph
with tf.InteractiveSession() as sess:
    # Initialize TensorFlow variablies
    sess.run(tf.global_variables_initializer)
    for i in range(1000):
        batch_xs, batch_ys = next_batch(100, data_train, label_train)
        sess.run([cost, train], feed_dict = {X: batch_xs, Y: batch_ys})
        
        
# Evaluation
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(label_train, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        


    
    
    
    
    
    
    
    
    


