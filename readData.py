import tensorflow as tf 
from openpyxl import load_workbook


# Read the data from xlsx file
path2 = "/Users/apple/Desktop/stuData.xlsx"     # Create the path of the file
workBook = load_workbook(path2)                 # Load the file
dataSheet = workBook.get_sheet_by_name('TrainData')  # Get the datasheet

# Range from the datasheet to assign the training set, test set and validation set.



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
