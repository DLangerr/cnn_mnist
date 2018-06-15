import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data
print("Downloading mnist data . . .")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_images = mnist.train.images
mnist_labels = mnist.train.labels
print("Finished downloading.")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

i = 0
def next_batch(size):
	global i
	if i*size+size > len(X_train):
		i = 0
	x_train_batch = X_train[size*i:i*size+size, :]
	y_train_batch = y_train_ohe[size*i:i*size+size, :]
	i = i + 1
	return x_train_batch, y_train_batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

def init_bias(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

keep_prob = tf.placeholder(tf.float32)
phX = tf.placeholder(tf.float32, [None, 784])
phY = tf.placeholder(tf.float32, [None, 10])


W1 = init_weights([3, 3, 1, 32])
b1 = init_bias([32])
W2 = init_weights([3, 3, 32, 64])
b2 = init_bias([64])

W3 = init_weights([3136, 1024])
b3 = init_bias([1024])
W4 = init_weights([1024, 512])
b4 = init_bias([512])
W5 = init_weights([512, 10])
b5 = init_bias([10])


X_image = tf.reshape(phX, [-1, 28, 28, 1])

h_conv1 = tf.nn.leaky_relu(conv2d(X_image, W1) + b1)
h_pool1 = max_pool(h_conv1)
h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool(h_conv2)

h_pool2 = tf.reshape(h_pool2, [-1, 3136])

    
h_fc_1 = tf.nn.leaky_relu(tf.matmul(h_pool2, W3) + b3)
h_fc_1 = tf.nn.dropout(h_fc_1, keep_prob)
h_fc_2 = tf.nn.leaky_relu(tf.matmul(h_fc_1, W4) + b4)
y_ = tf.matmul(h_fc_2, W5) + b5
pred = tf.nn.softmax(y_)

lr = 0.0001
epochs = 10000
if(len(sys.argv) == 2):
	epochs = int(sys.argv[1])

print(f"Running {epochs} training iterations . . .")
batch_size = 200

cross_entr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=phY, logits=y_))

train = tf.train.AdamOptimizer(lr).minimize(cross_entr)

init = tf.global_variables_initializer()

costs = []

with tf.Session() as sess:
    
    sess.run(init)
    
    for j in range(epochs):
        
        batch_X, batch_Y = mnist.train.next_batch(batch_size)
        
        sess.run(train, feed_dict={phX: batch_X, phY: batch_Y, keep_prob: 0.5})
        
        c = sess.run(cross_entr, feed_dict={phX:batch_X, phY:batch_Y, keep_prob: 0.5})
        costs.append(c)
        
        if j % (epochs//10) == 0:
           
            print(f"Iteration {j}. Cost {c}.")
            
    predictions = sess.run(pred, feed_dict={phX: mnist.test.images[:2000,:], keep_prob: 1.0})

argmax_pred = np.argmax(predictions, axis=1)
argmax_labels = np.argmax(mnist.test.labels[:2000,:], axis=1)
accuracy = np.mean(argmax_pred == argmax_labels)
print("==========================================")
print(f"Achieved accuracy of {accuracy*100}%.")

plt.plot(costs)
plt.title(f"Cost over {epochs} iterations.")
plt.show()
