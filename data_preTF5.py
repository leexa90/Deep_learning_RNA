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
#data = np.load('data_0nan.npy.zip')['data_0nan'].item()
data1 = np.load('data_484_nan.npy').item()
data2 = np.load('data_484_ss.npy').item()
data3 = np.load('data_484_extra.npy').item()
data = {}
'''
v5 - make 3 classes, <8 , >=8 & <=15 , >=15

'''
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
for i in data1.keys():
    if i in data2.keys():
        temp1 = data1[i]
        a,b,c = (data1[i][2] < 8)*1,(data1[i][2] <= 15) & (data1[i][2] >= 8)*1,(data1[i][2] > 15)*1
        temp_resi_map = np.stack((a,b,c),axis=2)
        d = -1*(np.isnan(data1[i][2])-1) #non-nan values ==1 , nan =0
        d = np.stack((d,d,d),axis=2)
        
        temp2 = data2[i]
        temp3 = data3[i]
        tempF = np.concatenate((np.array(make_array(temp1[1])).T,np.array([temp2[1]]),temp3))
        data[i] = [tempF, temp1[0],temp1[1],temp_resi_map,d,temp2[1]]
#np.save('data_all.npy',data)
die
data2_x = []
data2_y = []
data2_y_nan = []
data2_name = []
# Import MINST data
#from tensorflow.examples.tutorials.mnist import input_data
#st = input_data.read_data_sets("MNIST_data/", one_hot=True)



for i in data:
        data2_x += [data[i][0],]
        data2_y += [data[i][3],]
        data2_y_nan += [data[i][-2],]
        data2_name += [i,]
print (len(data2_y))
#data2_y = np.array(data2_y)
#data2_x = np.array(data2_x)
# Create some wrappers for simplicity
epsilon = 1e-3
def batch_normalization(x):
    mean,var = tf.nn.moments(x,[1,2])
    scale = tf.Variable(tf.ones([x.shape[-1]]))
    beta = tf.Variable(tf.zeros([x.shape[-1]]))
    x = tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon)
    return x
def conv2d(x, W, b, strides=1,F=False):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = batch_normalization(x)
    #x = tf.
    if F==False:
        return tf.nn.relu(x)
    else:
        return tf.nn.relu(x),x

# Parameters
learning_rate = 0.001
training_epochs = 300
batch_size = 100
display_step = 1
n_classes =5



# tf Graph input
x = tf.placeholder(tf.float32,[1,9,None,1])
# pairwise distances need to define
resi_map0 = tf.placeholder(tf.float32,[1,None,None,3])
# some values are na, these will be excluded from loss
above_zero = tf.placeholder(tf.float32,[1,None,None,3])
above_zero = tf.cast(above_zero,dtype=tf.float32) #TF to float
# if tf.is_nan, use convert to 0, else use original values
resi_map = tf.where(tf.is_nan(resi_map0),above_zero,resi_map0)
#resi_map = tf.reshape(resi_map, shape=[1, -1, -1, 1])
#x = tf.reshape(x, shape=[1, -1, 4, 1])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([1, 9, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([1, 9, 32, 64])),
    # 5x5 conv, 64 inputs, 64 outputs
    'wc3a': tf.Variable(tf.random_normal([1, 10, 64, 64])),
    'wc3b': tf.Variable(tf.random_normal([5, 1, 64, 64])),    
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes])),
    'outa': tf.Variable(tf.random_normal([5,5,64, 32])),
    'outb': tf.Variable(tf.random_normal([5,5,32, 16])),
    'outc': tf.Variable(tf.random_normal([5,5,16, 8])),
    'out2': tf.Variable(tf.random_normal([5,5,8, 3]))
    
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3a': tf.Variable(tf.random_normal([64])),
    'bc3b': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes])),
    'outa': tf.Variable(tf.random_normal([32])),
    'outb': tf.Variable(tf.random_normal([16])),
    'outc': tf.Variable(tf.random_normal([8])),
    'out2': tf.Variable(tf.random_normal([3]))
}

# construct model
norm_x = batch_normalization(x)
conv1 = conv2d(norm_x,weights['wc1'],biases['bc1'])
conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
final = []
for i in range(0,64):
    mat_x = conv2[:,:,:,i]
    final += [tf.matmul(mat_x,mat_x,transpose_a=True),]
    #final += tf.reshape(final[i],(-1,100,100,1))
y = tf.stack(final,axis=3)
y2 = batch_normalization(y)
conv3a = conv2d(y2,weights['wc3a'],biases['bc3a'])
conv3b = conv2d(conv3a,weights['wc3b'],biases['bc3b'])
outa = conv2d(conv3b,weights['outa'],biases['outa'])
outb = conv2d(outa,weights['outb'],biases['outb'])
outc = conv2d(outb,weights['outc'],biases['outc'])
out = conv2d(outc,weights['out2'],biases['out2'])
out_softmax = tf.nn.softmax(out,3,name='softmax')
# kill entries of nan so they are not in cost, not needed ???
#out_softmax2 = tf.multiply(out_softmax,above_zero,name='zero') 


# Define loss and optimizer
out_LogSoftmax = tf.log(out_softmax,'log')
log_loss = tf.multiply(above_zero,out_LogSoftmax,'logloss')
#rmse = tf.square(tf.subtract(out2,resi_map))     
cost = tf.multiply(tf.reduce_mean(out_LogSoftmax),-1)
#cost = tf.div(sum_logLoss,tf.reduce_sum(above_zero))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
##correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
##accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Training cycle
result = {}
for epoch in range(training_epochs):
    avg_cost = 0.
    val_cost = 0.
    total_batch = 400#int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(484):
        if i < 400:
            if i%100 == 99:
                print (i,400*avg_cost/(i+1))
            batch_x, batch_y = np.array([[data2_x[i],],]),np.array([data2_y[i],])
            batch_y_nan = np.array([data2_y_nan[i]])
            batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          resi_map0: batch_y,
                                                          above_zero : batch_y_nan})
            # Compute average loss
            avg_cost += c / total_batch
            #print (c),
        else:
            batch_x, batch_y = np.array([[data2_x[i],],]),np.array([data2_y[i],])
            batch_y_nan = np.array([data2_y_nan[i]])
            batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
            cost_i  = sess.run( cost, feed_dict={x: batch_x,resi_map0: batch_y,
                                                 above_zero : batch_y_nan})
            val_cost += cost_i/84
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch+1), "cost=", 
            "{:.9f}".format(avg_cost))
    if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch+1), "cost=", 
            "{:.9f}".format(val_cost))
    result[epoch] = [avg_cost,val_cost]
    pred = sess.run( out_softmax, feed_dict={x: batch_x,resi_map0: batch_y,above_zero : batch_y_nan})
##        print (pred[0,0:10,0:10,0])
##        print (batch_y[0,0:3,0:3,0])
##        print (sess.run( above_zero, feed_dict={x: batch_x,resi_map0: batch_y})[0,60:80,60:80,0])
##        print (sess.run( conv2, feed_dict={x: batch_x,resi_map0: batch_y})[0,0:10,0:10,0])
##        #print(sess.run( resi_map, feed_dict={x: batch_x,resi_map0: batch_y})[0,0:10,0:10,0])
##        print (sess.run( y, feed_dict={x: batch_x,resi_map0: batch_y})[0,0:10,0:10,0])
print ("Optimization Finished!")
save_path = saver.save(sess,'model300.ckpt')
plt.plot(range(0,300),[result[i][0] for i in result],label='Train')
plt.plot(range(0,300),[result[i][1] for i in result],label='Val')
plt.legend();plt.ylabel('RMSE cost') ; plt.xlabel('epoch')
plt.savefig('Train_curve.png')
result_cord = {}
for i in range(0,357):
	batch_x, batch_y = np.array([[data2_x[i],],]),np.array([[data2_y[i],],])
	batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
	batch_y = np.swapaxes(np.swapaxes(batch_y,1,3),1,2)
	result_cord[(i,data2_name[i])] = data[data2_name[i]] + [sess.run(out, feed_dict={x: batch_x,resi_map0: batch_y}),]
np.save('result_cord.npy',result_cord)
fig, ax = plt.subplots(2, sharex=True,figsize=(7, 15))
ax[1].imshow(sess.run( resi_map, feed_dict={x: batch_x,resi_map0: batch_y})[0,:,:,0])
ax[0].imshow(sess.run( out2, feed_dict={x: batch_x,resi_map0: batch_y})[0,:,:,0])
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
