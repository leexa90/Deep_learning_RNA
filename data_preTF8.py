from __future__ import print_function
from __future__ import absolute_import

import warnings
import sys
sys.path.append('../')
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
import random
data = np.load('../data_all.npy.zip')['data_all'].item()
data1 = np.load('../data_484_nan.npy').item()
data2 = np.load('../data_484_ss.npy').item()
data3 = np.load('../data_484_extra.npy').item()
data4 = np.load('../data_484_MSA.npy').item()

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
        elif str[i]=='a':
            temp[i] = [0.5,0,0,0,0]
        elif str[i] =='u':
            temp[i] = [0,0.5,0,0,0]
        elif str[i] =='g':
            temp[i] = [0,0,0.5,0,0]
        elif str[i] =='c':
            temp[i] = [0,0,0,0.5,0]
        else:
            temp[i] = [0,0,0,0,1]
    return temp
def make_array2(str):
    temp = [0,]*len(str)
    for i in range(len(str)):
        if str[i]=='*':
            temp[i] = 100
        elif str[i] == 'X':
            temp[i] = 0
        else:
            temp[i] = int(str[i])*10
    return temp
data1_keys = data1.keys()
random.shuffle(data1_keys)
for i in data1_keys:
    if len(data1[i][0]) >= 35:
        if i in data2.keys():
            temp1 = data1[i]
            a,b,c = (data1[i][2] < 8)*1,(data1[i][2] <= 15) & (data1[i][2] >= 8)*1,(data1[i][2] > 15)*1
            temp_resi_map = np.stack((a,b,c),axis=2)
            d = -1*(np.isnan(data1[i][2])-1) #non-nan values ==1 , nan =0
            d = np.stack((d,d,d),axis=2)
            
            temp2 = data2[i]
            temp3 = data3[i]
            temp4 = data4[i]
            tempF = np.concatenate((np.array(make_array(temp1[1])).T,np.array([temp2[1]]),temp3,(np.array(make_array(temp4[1])).T),np.array([make_array2(temp4[2])])))
            #         [9-features, seq, exxist_seq, cat dist_map,cat dist_map (non-zero), ss_1d, ss_2d] 
            data[i] = [tempF, temp1[0],temp1[1],temp_resi_map,d,temp2[1],temp2[0]]
#np.save('data_all.npy',data)

data2_x = []
data2_y = []
data2_y_nan = []
data2_y_ss = []
data2_name = []
# Import MINST data
#from tensorflow.examples.tutorials.mnist import input_data
#st = input_data.read_data_sets("MNIST_data/", one_hot=True)



for i in data:
        data2_x += [data[i][0],]
        data2_y += [data[i][3],]
        data2_y_nan += [data[i][-3],]
        data2_y_ss += [data[i][-1],]
        data2_name += [i,]
print (len(data2_y),'number of training samples')
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
def conv2d(x, W, b, strides=1,relu=True,padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    x = batch_normalization(x)
    #x = tf.
    if relu is True:
        return tf.nn.relu(x)
    else:
        return x

# Parameters
learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1
n_classes =5



# tf Graph input
x = tf.placeholder(tf.float32,[1,15,None,1])
# pairwise distances need to define
resi_map0 = tf.placeholder(tf.float32,[1,None,None,3])
# some values are na, these will be excluded from loss
above_zero = tf.placeholder(tf.float32,[1,None,None,3])
above_zero = tf.cast(above_zero,dtype=tf.float32) #TF to float
# if tf.is_nan, use convert to 0, else use original values
resi_map = tf.where(tf.is_nan(resi_map0),above_zero,resi_map0)
# ss_2d
ss_2d = tf.placeholder(tf.float32,[1,None,None])
#resi_map = tf.reshape(resi_map, shape=[1, -1, -1, 1])
#x = tf.reshape(x, shape=[1, -1, 4, 1])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    '1_wc1': tf.Variable(tf.random_normal([15, 1, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    '2_wc2': tf.Variable(tf.random_normal([1, 15, 32, 63])),
    # 5x5 conv, 64 inputs, 64 outputs
    '3_wc3a': tf.Variable(tf.random_normal([1, 10, 64, 32])),
    '4_wc3b': tf.Variable(tf.random_normal([10, 1, 64, 32])),    
    # 1024 inputs, 10 outputs (class prediction)
    '5_outaa': tf.Variable(tf.random_normal([10,1,64, 16])),
    '6_outab': tf.Variable(tf.random_normal([1,10,64, 16])),
    '7_outb': tf.Variable(tf.random_normal([5,5,32, 16])),
    '8_outc': tf.Variable(tf.random_normal([5,5,16, 8])),
    '9_out2': tf.Variable(tf.random_normal([5,5,8, 3]))   
}
print ('number of weights')
total_parameters = 0
for variable in weights:
    # shape is an array of tf.Dimension
    shape = weights[variable].get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    print(variable,variable_parameters)
    total_parameters += variable_parameters
print(total_parameters, ':total parameters')

biases = {
    '1_bc1': tf.Variable(tf.random_normal([32])),
    '2_bc2': tf.Variable(tf.random_normal([63])),
    '3_bc3a': tf.Variable(tf.random_normal([32])),
    '4_bc3b': tf.Variable(tf.random_normal([32])),
    '5_outaa': tf.Variable(tf.random_normal([16])),
    '6_outab': tf.Variable(tf.random_normal([16])),
    '7_outb': tf.Variable(tf.random_normal([16])),
    '8_outc': tf.Variable(tf.random_normal([8])),
    '9_out2': tf.Variable(tf.random_normal([3]))
}

# construct model
norm_x = batch_normalization(x)
conv1 = conv2d(norm_x,weights['1_wc1'],biases['1_bc1'],padding='VALID')
conv2 = conv2d(conv1,weights['2_wc2'],biases['2_bc2'])
final = []
for i in range(0,63):
    mat_x = conv2[:,:,:,i]
    final += [tf.matmul(mat_x,mat_x,transpose_a=True),]
    #final += tf.reshape(final[i],(-1,100,100,1))
final +=  [ss_2d,]
y = tf.stack(final,axis=3)
y2 = batch_normalization(y)
conv3aa = conv2d(y2,weights['3_wc3a'],biases['3_bc3a'])
conv3ab = conv2d(y2,weights['4_wc3b'],biases['4_bc3b'])
conv3b = tf.concat([conv3aa,conv3ab],axis=3)
conv3bp = tf.layers.average_pooling2d(conv3ab,(2,2,),1)
outaa = conv2d(conv3bp,weights['5_outaa'],biases['5_outaa'])
outab = conv2d(conv3bp,weights['6_outab'],biases['6_outab'])
outa = tf.concat([outaa,outab],axis=3)
outa_p = tf.layers.average_pooling2d(outa,(2,2,),1)
outb = conv2d(outa,weights['7_outb'],biases['7_outb'])
outb_p = tf.layers.average_pooling2d(outb,(2,2,),1)
outc = conv2d(outb_p,weights['8_outc'],biases['8_outc'])
out = conv2d(outc,weights['9_out2'],biases['9_out2'],relu=False)
out_softmax = tf.nn.softmax(out,-1,name='softmax')
out_norm = batch_normalization(out)
# kill entries of nan so they are not in cost, not needed ???
#out_softmax2 = tf.multiply(out_softmax,above_zero,name='zero') 


# Define loss and optimizer

log_loss =tf.nn.softmax_cross_entropy_with_logits(logits = out_norm , labels = resi_map ,dim=-1)
cost = tf.reduce_mean(log_loss)
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
train_n = len(data2_x)*0.7//1
val_n = len(data2_x) - train_n
print ('training samples %s , val samples %s' %(train_n,val_n))
for epoch in range(training_epochs):
    avg_cost = 0.
    val_cost = 0.
    total_batch = 400#int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(len(data2_x)):
        if i < train_n:
            if i%100 == 99:
                print (i,train_n*avg_cost/(i+1))
            batch_x, batch_y = np.array([[data2_x[i],],]),np.array([data2_y[i],])
            batch_y_nan,batch_y_ss = np.array([data2_y_nan[i]]),np.array([data2_y_ss[i]])
            batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          resi_map0: batch_y,
                                                          above_zero : batch_y_nan,
                                                          ss_2d : batch_y_ss})
            # Compute average loss
            avg_cost += c / total_batch
            #print (c),
        else:
            batch_x, batch_y = np.array([[data2_x[i],],]),np.array([data2_y[i],])
            batch_y_nan,batch_y_ss = np.array([data2_y_nan[i]]),np.array([data2_y_ss[i]])
            batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
            cost_i  = sess.run( cost, feed_dict={x: batch_x,resi_map0: batch_y,
                                                 above_zero : batch_y_nan, ss_2d : batch_y_ss})
            val_cost += cost_i/val_n
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch+1), "cost=", 
            "{:.9f}".format(avg_cost))
    if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch+1), "cost=", 
            "{:.9f}".format(val_cost))
    result[epoch] = [avg_cost,val_cost]
    pred = sess.run( out_softmax, feed_dict={x: batch_x,resi_map0: batch_y,
                                             above_zero : batch_y_nan, ss_2d : batch_y_ss})
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
plt.legend();plt.ylabel('Logloss cost') ; plt.xlabel('epoch')
plt.savefig('Train_curveLg.png')
result_cord = {}
for i in range(484):
        batch_x, batch_y = np.array([[data2_x[i],],]),np.array([data2_y[i],])
        batch_y_nan,batch_y_ss = np.array([data2_y_nan[i]]),np.array([data2_y_ss[i]])
        batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
        result_cord[(i,data2_name[i])] = data[data2_name[i]] + [sess.run(out, feed_dict={x: batch_x,resi_map0: batch_y,
                                                                                         above_zero : batch_y_nan, ss_2d : batch_y_ss}),]
np.save('result_cord.npy',result_cord)
fig, ax = plt.subplots(2, sharex=True,figsize=(7, 15))
ax[1].imshow(sess.run( resi_map, feed_dict={x: batch_x,resi_map0: batch_y,above_zero : batch_y_nan, ss_2d : batch_y_ss})[0,:,:,0])
ax[0].imshow(sess.run( out2, feed_dict={x: batch_x,resi_map0: batch_y,above_zero : batch_y_nan, ss_2d : batch_y_ss})[0,:,:,0])
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
