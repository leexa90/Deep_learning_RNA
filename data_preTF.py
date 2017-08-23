from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input, Lambda
from keras.layers import Flatten, Reshape, Dropout, merge
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data = np.load('DATA.npy').item()
data2_x = []
data2_y = []
data2_name = []
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def make_array(str):
    temp = [0,]*len(str)
    for i in range(len(str)):
        if str[i]=='A':
            temp[i] = [1,0,0,0,0]
        elif str[i] =='U':
            temp[i] = [0,1,0,0,0]
        elif str[i] =='G':
            temp[i] = [0,0,1,0,0]
        elif str[i] =='C':
            temp[i] = [0,0,0,1,0]
        else:
            temp[i] = [0,0,0,0,1]
    return temp

for i in sorted(data):
    if len(data[i][0]) < 100:
        data2_x += [make_array(data[i][1]),]
        data2_y += [data[i][2],]
        data2_name += [i,]
print (len(data2_y))
#data2_y = np.array(data2_y)
#data2_x = np.array(data2_x)
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
n_classes =5



# tf Graph input
x = tf.placeholder(tf.float32,[1,None,5,1])
resi_map = tf.placeholder(tf.float32,[1,None,None,1])
not_zero = tf.not_equal(resi_map,0)
#resi_map = tf.reshape(resi_map, shape=[1, -1, -1, 1])
#x = tf.reshape(x, shape=[1, -1, 4, 1])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([10, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([10, 5, 32, 64])),
    # 5x5 conv, 64 inputs, 64 outputs
    'wc3a': tf.Variable(tf.random_normal([10, 1, 64, 64])),
    'wc3b': tf.Variable(tf.random_normal([1, 5, 64, 64])),    
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes])),
    'out2': tf.Variable(tf.random_normal([5,5,64, 1]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3a': tf.Variable(tf.random_normal([64])),
    'bc3b': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes])),
    'out2': tf.Variable(tf.random_normal([1]))
}

# construct model
conv1 = conv2d(x,weights['wc1'],biases['bc1'])
final = []
for i in range(0,32):
    mat_x = conv1[:,:,:,1]
    final += [tf.matmul(mat_x,mat_x,transpose_b=True),]
    #final += tf.reshape(final[i],(-1,100,100,1))
y = tf.stack(final,axis=3)
conv2 = conv2d(y,weights['wc2'],biases['bc2'])
conv3a = conv2d(conv2,weights['wc3a'],biases['bc3a'])
conv3b = conv2d(conv3a,weights['wc3b'],biases['bc3b'])
out = conv2d(conv3b,weights['out2'],biases['out2'])
#out = tf.reshape(out,shape=(-1,100,100))

# Define loss and optimizer
cost = tf.reduce_mean([out,resi_map])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
##correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
##accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(300):
            if i%100 == 0:
                print (i)
            batch_x, batch_y = np.array([[data2_x[i],],]),np.array([[data2_y[i],],])
            batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
            batch_y = np.swapaxes(np.swapaxes(batch_y,1,3),1,2)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          resi_map: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", 
                "{:.9f}".format(avg_cost))
    print ("Optimization Finished!")

die
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

seq_input = Input(shape=(100,5,1))

ss_input = Input(shape=(100,100,1))

x = conv2d_bn(seq_input, 32, 10, 5, strides=(1, 1), padding='same')
#x = conv2d_bn(x, 32, 3, 3, strides=(1, 1), padding='same')
#x = conv2d_bn(x, 32, 3, 3, strides=(1, 1), padding='same')
def multiply(x,n):
    x_prime = tf.reshape(x, (-1, n, 1))
    x_transpose = tf.transpose(x_prime, perm=[0,2, 1])
    return tf.matmul(x_transpose,x_prime)


#Lambda(lambda x: multiply(x, n), output_shape =(n, n))

# Input is 100 * 5 matrix
seq_input = Input(shape=(100,5))

# convert to tensor and get 10 layers
x = layers.Reshape((100,5,1))(seq_input)
x = Conv2D( filters =10 , kernel_size = (3,3),strides=(1, 1),padding='same') ( x)

# get outer product to get 100*100 matrix for each layer
final = {}
def matmul(mat_x):
    y = K.tf.matmul( mat_x, mat_x, transpose_b=True )
    return y
def multiply(x,n=100):
    x_prime = tf.reshape(x, (-1, n, 5))
    x_transpose = tf.transpose(x_prime, perm=[0,2, 1])
    return tf.matmul(x_prime,x_transpose)
for i in range(0,10):
    mat_x = x[:,:,:,i]
    final[i] = Lambda(lambda x: multiply(x, n=100) , output_shape = (100,100))(mat_x)#Lambda( matmul,output_shape= (-1,100, 100,1) ) (mat_x)
    #final[i] =  K.dot(mat_x,K.permute_dimensions(mat_x,(0,2,1)))
    final[i] = K.reshape(final[i],(-1,100,100,1))
    
y = merge([ final[idx] for idx in final],mode='concat', concat_axis=3)
#y = Reshape((100,100,10))(y)
z = Activation('relu')(y)
model = Model ([seq_input,ss_input],z)

import tensorflow as tf
sess = K.get_session()
q=K.eval
from keras import backend as K
#K.set_session(sess)
with sess.as_default():
    x = [[1,1],[3,4],[5,6]]
    z = tf.Variable(x)
    z2 = K.reshape(z,(6,2))
    z.eval()
    die

