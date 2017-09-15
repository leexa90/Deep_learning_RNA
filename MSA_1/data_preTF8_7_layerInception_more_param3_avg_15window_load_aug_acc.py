from __future__ import print_function
from __future__ import absolute_import

import warnings
import sys
sys.path.append('../../../../')
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
#data = np.load('../../../../data_all.npy.zip')['data_all'].item()
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
random.seed(0)
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
            for window_tup in [(35,1),(50,2),(75,3),(100,4),(150,6),(200,8),(400,16)]:
                window, jump = window_tup[0], window_tup[1]
                for repeat in range(0,len(data1[i][0]) - window+1,jump):
                    data[i+'_'+str(window)+'_'+str(repeat)] = [tempF[:,repeat:repeat+window],
                                           temp1[0][repeat:repeat+window],
                                           temp1[1][repeat:repeat+window],
                                           temp_resi_map[repeat:repeat+window,repeat:repeat+window,:],
                                           d[repeat:repeat+window,repeat:repeat+window,:],
                                           temp2[1][repeat:repeat+window],
                                           temp2[0][repeat:repeat+window,repeat:repeat+window]]
            
#np.save('data_all.npy',data)
dictt = {}
for i in data.keys():
	if tuple(i.split('_')[0:2]) not in dictt :
		dictt[tuple(i.split('_')[0:2])] = 1
	else :
		dictt[tuple(i.split('_')[0:2])] += 1
#plt.hist([dictt[x] for x in dictt],100);plt.show()
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
def conv2d(x, W, b, strides=(1,1),relu=True,padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides[0], strides[0], 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    x = batch_normalization(x)
    #x = tf.
    if relu is True:
        return tf.nn.relu(x)
    else:
        return x
def average_pooling2d(x,window = (2,2),strides=1,padding='same'):
    x = tf.layers.average_pooling2d(x,window,strides,padding=padding)
    x = batch_normalization (x)
    return tf.nn.relu(x)
# Parameters
learning_rate = 0.001
training_epochs = 40 
batch_size = 1
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
window = 15
num1 = 32/4
num2 = 64/4
num3 = 64/4
num4 = 64/4
num5 = 32/4
num6 = 32/4
num7 = 16/4
# Store layers weight & bias
weights = {
    # 1D inception layer
    '1_wc1aa': tf.Variable(tf.random_normal([15, 1, 1, num1])),
    '1_wc1ab': tf.Variable(tf.random_normal([1, window, num1, num1])),
    '1_wc1ac': tf.Variable(tf.random_normal([1, window, num1, num1])),
    '1_wc1ba': tf.Variable(tf.random_normal([15, 1, 1, num1])),
    '1_wc1bb': tf.Variable(tf.random_normal([1, window, num1, num1])),
    '1_wc1c': tf.Variable(tf.random_normal([15, 1, 1, num1])),
    '1_wc1d': tf.Variable(tf.random_normal([15, 1, 1, num1])),
    # 1D inception layer
    '2_wc1aa': tf.Variable(tf.random_normal([1, 1, num1*4, num2])),
    '2_wc1ab': tf.Variable(tf.random_normal([1, window, num2, num2])),
    '2_wc1ac': tf.Variable(tf.random_normal([1, window, num2, num2])),
    '2_wc1ba': tf.Variable(tf.random_normal([1, 1, num1*4, num2])),
    '2_wc1bb': tf.Variable(tf.random_normal([1, window, num2, num2])),
    '2_wc1c': tf.Variable(tf.random_normal([1, 1, num1*4,num2])),
    '2_wc1d': tf.Variable(tf.random_normal([1, 1, num1*4, num2])),
    
    # 2D inception layer output 96 layer
    '3_wc1aa': tf.Variable(tf.random_normal([1, 1, num2*4+1, num3])),
    '3_wc1ab': tf.Variable(tf.random_normal([window, 1, num3, num3])),
    '3_wc1ac': tf.Variable(tf.random_normal([1, window, num3, num3])),
    '3_wc1ba': tf.Variable(tf.random_normal([1, 1, num2*4+1, num3])),
    '3_wc1bb': tf.Variable(tf.random_normal([window/2, window/2, num3, num3])),
    '3_wc1c': tf.Variable(tf.random_normal([1, 1, num2*4+1, num3])),
    '3_wc1d': tf.Variable(tf.random_normal([1, 1, num2*4+1, num3])),
    
##    '4_wc1aa': tf.Variable(tf.random_normal([1, 1, num3*4, 16])),
##    '4_wc1ab': tf.Variable(tf.random_normal([window, window, 16, 16])),
##    '4_wc1ac': tf.Variable(tf.random_normal([window, window, 16, 16])),
##    '4_wc1ba': tf.Variable(tf.random_normal([1, 1, num3*4, 16])),
##    '4_wc1bb': tf.Variable(tf.random_normal([window, window, 16, 16])),
##    '4_wc1c': tf.Variable(tf.random_normal([1, 1, num3*4, 16])),
##    '4_wc1d': tf.Variable(tf.random_normal([1, 1, num3*4, 16])),

    '4_wc1aa': tf.Variable(tf.random_normal([1, 1, num3*4, num4])),
    '4_wc1ab': tf.Variable(tf.random_normal([1, window, num4, num4])),
    '4_wc1ac': tf.Variable(tf.random_normal([window, 1, num4, num4])),
    '4_wc1ba': tf.Variable(tf.random_normal([1, 1, num3*4, num4])),
    '4_wc1bb': tf.Variable(tf.random_normal([window/2, window/2, num4, num4])),
    '4_wc1c': tf.Variable(tf.random_normal([1, 1, num3*4, num4])),
    '4_wc1d': tf.Variable(tf.random_normal([1, 1, num3*4, num4])),

    '5_wc1aa': tf.Variable(tf.random_normal([1, 1, num4*4, num5])),
    '5_wc1ab': tf.Variable(tf.random_normal([window, 1, num5, num5])),
    '5_wc1ac': tf.Variable(tf.random_normal([1, window, num5, num5])),
    '5_wc1ba': tf.Variable(tf.random_normal([1, 1, num4*4, num5])),
    '5_wc1bb': tf.Variable(tf.random_normal([window/2, window/2, num5, num5])),
    '5_wc1c': tf.Variable(tf.random_normal([1, 1, num4*4, num5])),
    '5_wc1d': tf.Variable(tf.random_normal([1, 1, num4*4, num5])),

    '6_wc1aa': tf.Variable(tf.random_normal([1, 1, num5*4, num6])),
    '6_wc1ab': tf.Variable(tf.random_normal([1, window, num6, num6])),
    '6_wc1ac': tf.Variable(tf.random_normal([window, 1, num6, num6])),
    '6_wc1ba': tf.Variable(tf.random_normal([1, 1, num5*4, num6])),
    '6_wc1bb': tf.Variable(tf.random_normal([window/2, window/2, num6, num6])),
    '6_wc1c': tf.Variable(tf.random_normal([1, 1, num5*4, num6])),
    '6_wc1d': tf.Variable(tf.random_normal([1, 1, num5*4, num6])),

    '7_wc1aa': tf.Variable(tf.random_normal([1, 1, num6*4, num7])),
    '7_wc1ab': tf.Variable(tf.random_normal([window, 1, num7, num7])),
    '7_wc1ac': tf.Variable(tf.random_normal([1, window, num7, num7])),
    '7_wc1ba': tf.Variable(tf.random_normal([1, 1, num6*4, num7])),
    '7_wc1bb': tf.Variable(tf.random_normal([window/2, window/2, num7, num7])),
    '7_wc1c': tf.Variable(tf.random_normal([1, 1, num6*4, num7])),
    '7_wc1d': tf.Variable(tf.random_normal([1, 1, num6*4, num7])),

    '9_out2': tf.Variable(tf.random_normal([5,5,num4*4, 3]))   
}

biases = {
    '1_bc1aa': tf.Variable(tf.random_normal([num1])),
    '1_bc1ab': tf.Variable(tf.random_normal([num1])),
    '1_bc1ac': tf.Variable(tf.random_normal([num1])),
    '1_bc1ba': tf.Variable(tf.random_normal([num1])),
    '1_bc1bb': tf.Variable(tf.random_normal([num1])),
    '1_bc1c': tf.Variable(tf.random_normal([num1])),
    '1_bc1d': tf.Variable(tf.random_normal([num1])),

    '2_bc1aa': tf.Variable(tf.random_normal([num2])),
    '2_bc1ab': tf.Variable(tf.random_normal([num2])),
    '2_bc1ac': tf.Variable(tf.random_normal([num2])),
    '2_bc1ba': tf.Variable(tf.random_normal([num2])),
    '2_bc1bb': tf.Variable(tf.random_normal([num2])),
    '2_bc1c': tf.Variable(tf.random_normal([num2])),
    '2_bc1d': tf.Variable(tf.random_normal([num2])),

    '3_bc1aa': tf.Variable(tf.random_normal([num3])),
    '3_bc1ab': tf.Variable(tf.random_normal([num3])),
    '3_bc1ac': tf.Variable(tf.random_normal([num3])),
    '3_bc1ba': tf.Variable(tf.random_normal([num3])),
    '3_bc1bb': tf.Variable(tf.random_normal([num3])),
    '3_bc1c': tf.Variable(tf.random_normal([num3])),
    '3_bc1d': tf.Variable(tf.random_normal([num3])),

    '4_bc1aa': tf.Variable(tf.random_normal([num4])),
    '4_bc1ab': tf.Variable(tf.random_normal([num4])),
    '4_bc1ac': tf.Variable(tf.random_normal([num4])),
    '4_bc1ba': tf.Variable(tf.random_normal([num4])),
    '4_bc1bb': tf.Variable(tf.random_normal([num4])),
    '4_bc1c': tf.Variable(tf.random_normal([num4])),
    '4_bc1d': tf.Variable(tf.random_normal([num4])),

    '5_bc1aa': tf.Variable(tf.random_normal([num5])),
    '5_bc1ab': tf.Variable(tf.random_normal([num5])),
    '5_bc1ac': tf.Variable(tf.random_normal([num5])),
    '5_bc1ba': tf.Variable(tf.random_normal([num5])),
    '5_bc1bb': tf.Variable(tf.random_normal([num5])),
    '5_bc1c': tf.Variable(tf.random_normal([num5])),
    '5_bc1d': tf.Variable(tf.random_normal([num5])),

    '6_bc1aa': tf.Variable(tf.random_normal([num6])),
    '6_bc1ab': tf.Variable(tf.random_normal([num6])),
    '6_bc1ac': tf.Variable(tf.random_normal([num6])),
    '6_bc1ba': tf.Variable(tf.random_normal([num6])),
    '6_bc1bb': tf.Variable(tf.random_normal([num6])),
    '6_bc1c': tf.Variable(tf.random_normal([num6])),
    '6_bc1d': tf.Variable(tf.random_normal([num6])),

    '7_bc1aa': tf.Variable(tf.random_normal([num7])),
    '7_bc1ab': tf.Variable(tf.random_normal([num7])),
    '7_bc1ac': tf.Variable(tf.random_normal([num7])),
    '7_bc1ba': tf.Variable(tf.random_normal([num7])),
    '7_bc1bb': tf.Variable(tf.random_normal([num7])),
    '7_bc1c': tf.Variable(tf.random_normal([num7])),
    '7_bc1d': tf.Variable(tf.random_normal([num7])),
    
    '9_out2': tf.Variable(tf.random_normal([3]))
}

print ('number of weights')
total_parameters = 0
for variable in sorted(weights):
    # shape is an array of tf.Dimension
    shape = weights[variable].get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    print(variable,variable_parameters)
    total_parameters += variable_parameters
print(total_parameters, ':total parameters')



# construct model
# 1D first inception layer
norm_x = batch_normalization(x)
conv1aa = conv2d(norm_x,weights['1_wc1aa'],biases['1_bc1aa'],padding='VALID')
conv1ab = conv2d(conv1aa,weights['1_wc1ab'],biases['1_bc1ab'])
conv1ac = conv2d(conv1ab,weights['1_wc1ac'],biases['1_bc1ac'])
conv1ba = conv2d(norm_x,weights['1_wc1ba'],biases['1_bc1ba'],padding='VALID')
conv1bb = conv2d(conv1ba,weights['1_wc1bb'],biases['1_bc1bb'])
conv1ca = conv2d(norm_x,weights['1_wc1c'],biases['1_bc1c'],padding='VALID')
conv1cb = average_pooling2d(conv1ca,(1,2),1,padding='same')
conv1da = conv2d(norm_x,weights['1_wc1d'],biases['1_bc1d'],padding='VALID')
conv1p = tf.concat([conv1ac,conv1bb,conv1cb,conv1da],3)
conv1 =  average_pooling2d(conv1p ,  (1,2),1,padding='same')

# 1d second inception layer
conv2aa = conv2d(conv1,weights['2_wc1aa'],biases['2_bc1aa'])
conv2ab = conv2d(conv2aa,weights['2_wc1ab'],biases['2_bc1ab'])
conv2ac = conv2d(conv2ab,weights['2_wc1ac'],biases['2_bc1ac'])
conv2ba = conv2d(conv1,weights['2_wc1ba'],biases['2_bc1ba'])
conv2bb = conv2d(conv2ba,weights['2_wc1bb'],biases['2_bc1bb'])
conv2ca = average_pooling2d(conv1,(1,2),1,padding='same')
conv2cb = conv2d(conv1,weights['2_wc1c'],biases['2_bc1c'])
conv2da = conv2d(conv1,weights['2_wc1d'],biases['2_bc1d']) #not to confuse name with function
conv2p = tf.concat([conv2ac,conv2bb,conv2cb,conv2da],3)
conv2 =  average_pooling2d(conv2p ,  (1,2),1,padding='same')
final = []
for i in range(0,num2*4):
    mat_x = conv2[:,:,:,i]
    final += [tf.matmul(mat_x,mat_x,transpose_a=True),]
    #final += tf.reshape(final[i],(-1,100,100,1))
final +=  [ss_2d,]
y = tf.stack(final,axis=3)
y2 = batch_normalization(y)

# 2d first inception layer
conv3aa = conv2d(y2,weights['3_wc1aa'],biases['3_bc1aa'])
conv3ab = conv2d(conv3aa,weights['3_wc1ab'],biases['3_bc1ab'])
conv3ac = conv2d(conv3ab,weights['3_wc1ac'],biases['3_bc1ac'])
conv3ba = conv2d(y2,weights['3_wc1ba'],biases['3_bc1ba'])
conv3bb = conv2d(conv3ba,weights['3_wc1bb'],biases['3_bc1bb'])
conv3ca = average_pooling2d(y2,(2,2),1,padding='same')
conv3cb = conv2d(y2,weights['3_wc1c'],biases['3_bc1c'])
conv3d = conv2d(y2,weights['3_wc1d'],biases['3_bc1d'])
conv3p = tf.concat([conv3ac,conv3bb,conv3cb,conv3d],3)
conv3 =  average_pooling2d(conv3p ,  (2,2),1,padding='same')
# 2d second inception layer
conv4aa = conv2d(conv3,weights['4_wc1aa'],biases['4_bc1aa'])
conv4ab = conv2d(conv4aa,weights['4_wc1ab'],biases['4_bc1ab'])
conv4ac = conv2d(conv4ab,weights['4_wc1ac'],biases['4_bc1ac'])
conv4ba = conv2d(conv3,weights['4_wc1ba'],biases['4_bc1ba'])
conv4bb = conv2d(conv4ba,weights['4_wc1bb'],biases['4_bc1bb'])
conv4ca = average_pooling2d(conv3,(2,2),1,padding='same')
conv4cb = conv2d(conv3,weights['4_wc1c'],biases['4_bc1c'])
conv4d = conv2d(conv3,weights['4_wc1d'],biases['4_bc1d'])
conv4p = tf.concat([conv4ac,conv4bb,conv4cb,conv4d],3)
conv4 =  average_pooling2d(conv4p ,  (2,2),1,padding='same')
### 2d third inception layer
conv5aa = conv2d(conv4,weights['5_wc1aa'],biases['5_bc1aa'])
conv5ab = conv2d(conv5aa,weights['5_wc1ab'],biases['5_bc1ab'])
conv5ac = conv2d(conv5ab,weights['5_wc1ac'],biases['5_bc1ac'])
conv5ba = conv2d(conv4,weights['5_wc1ba'],biases['5_bc1ba'])
conv5bb = conv2d(conv5ba,weights['5_wc1bb'],biases['5_bc1bb'])
conv5ca = average_pooling2d(conv4,(2,2),1,padding='same')
conv5cb = conv2d(conv4,weights['5_wc1c'],biases['5_bc1c'])
conv5d = conv2d(conv4,weights['5_wc1d'],biases['5_bc1d'])
conv5p = tf.concat([conv5ac,conv5bb,conv5cb,conv5d],3)
conv5 =  average_pooling2d(conv5p ,  (2,2),1,padding='same')
# 2d third inception layer
conv6aa = conv2d(conv5,weights['6_wc1aa'],biases['6_bc1aa'])
conv6ab = conv2d(conv6aa,weights['6_wc1ab'],biases['6_bc1ab'])
conv6ac = conv2d(conv6ab,weights['6_wc1ac'],biases['6_bc1ac'])
conv6ba = conv2d(conv5,weights['6_wc1ba'],biases['6_bc1ba'])
conv6bb = conv2d(conv6ba,weights['6_wc1bb'],biases['6_bc1bb'])
conv6ca = average_pooling2d(conv5,(2,2),1,padding='same')
conv6cb = conv2d(conv5,weights['6_wc1c'],biases['6_bc1c'])
conv6d = conv2d(conv5,weights['6_wc1d'],biases['6_bc1d'])
conv6p = tf.concat([conv6ac,conv6bb,conv6cb,conv6d],3)
conv6 =  average_pooling2d(conv6p ,  (2,2),1,padding='same')
# 2d third inception layer
conv7aa = conv2d(conv6,weights['7_wc1aa'],biases['7_bc1aa'])
conv7ab = conv2d(conv7aa,weights['7_wc1ab'],biases['7_bc1ab'])
conv7ac = conv2d(conv7ab,weights['7_wc1ac'],biases['7_bc1ac'])
conv7ba = conv2d(conv6,weights['7_wc1ba'],biases['7_bc1ba'])
conv7bb = conv2d(conv7ba,weights['7_wc1bb'],biases['7_bc1bb'])
conv7ca = average_pooling2d(conv6,(2,2),1,padding='same')
conv7cb = conv2d(conv6,weights['7_wc1c'],biases['7_bc1c'])
conv7d = conv2d(conv6,weights['7_wc1d'],biases['7_bc1d'])
conv7 = tf.concat([conv7ac,conv7bb,conv7cb,conv7d],3)

out = conv2d(conv4,weights['9_out2'],biases['9_out2'],relu=False)
out_softmax = tf.nn.softmax(out,-1,name='softmax')
out_norm = batch_normalization(out)
# kill entries of nan so they are not in cost, not needed ???
#out_softmax2 = tf.multiply(out_softmax,above_zero,name='zero') 


# Define loss and optimizer

log_loss =tf.nn.softmax_cross_entropy_with_logits(logits = out_norm , labels = resi_map ,dim=-1)
cost = tf.reduce_mean(log_loss)
#cost = tf.div(sum_logLoss,tf.reduce_sum(above_zero))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
def accuracy(mat_model,answer):
    score = []
    for i in range(0,answer.shape[1]):
        for j in range(i+1,answer.shape[1]):
            if answer[i,j] == mat_model[i,j]:
                score += [1,]
            else:
                score += [0,]
    return np.mean(score)
# Evaluate model
##correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
##accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#saver.restore(sess,'model300.ckpt')
# Training cycle
result = {}
train_n = len(data2_x)*0.7//1
val_n = len(data2_x) - train_n
print ('training samples %s , val samples %s' %(train_n,val_n))
random.seed(0)
shuffle = range(len(data2_x))
random.shuffle(shuffle)
for epoch in range(training_epochs):
    avg_cost = 0.
    val_cost = 0.
    train_acc = []
    val_acc = []
    total_batch = train_n#int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in shuffle:
        if i < train_n:
            if i%100 == 9999:
                print (i,train_n*avg_cost/(i+1))
            batch_x, batch_y = np.array([[data2_x[i],],]),np.array([data2_y[i],])
            batch_y_nan,batch_y_ss = np.array([data2_y_nan[i]]),np.array([data2_y_ss[i]])
            batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          resi_map0: batch_y,
                                                          above_zero : batch_y_nan,
                                                          ss_2d : batch_y_ss})
            pred =sess.run( out_softmax, feed_dict={x: batch_x,resi_map0: batch_y,
                                                 above_zero : batch_y_nan, ss_2d : batch_y_ss})
            train_acc += [accuracy(np.argmax(pred,3)[0],np.argmax(batch_y,3)[0]),]
            # Compute average loss
            avg_cost += c / total_batch
            #print (train_acc)
            #print (c),
        else:
            batch_x, batch_y = np.array([[data2_x[i],],]),np.array([data2_y[i],])
            batch_y_nan,batch_y_ss = np.array([data2_y_nan[i]]),np.array([data2_y_ss[i]])
            batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
            cost_i  = sess.run( cost, feed_dict={x: batch_x,resi_map0: batch_y,
                                                 above_zero : batch_y_nan, ss_2d : batch_y_ss})
            val_cost += cost_i/val_n
            pred =sess.run( out_softmax, feed_dict={x: batch_x,resi_map0: batch_y,
                                                 above_zero : batch_y_nan, ss_2d : batch_y_ss})
            val_acc += [accuracy(np.argmax(pred,3)[0],np.argmax(batch_y,3)[0]),]
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch+1), "cost=", 
            "{:.9f}".format(avg_cost),np.mean(train_acc))
    if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch+1), "cost=", 
            "{:.9f}".format(val_cost),np.mean(val_acc))
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
plt.plot(range(0,training_epochs),[result[i][0] for i in result],label='Train')
plt.plot(range(0,training_epochs),[result[i][1] for i in result],label='Val')
plt.legend();plt.ylabel('Logloss cost') ; plt.xlabel('epoch')
plt.savefig('Train_curveLg.png')
result_cord = {}
for i in range(482):
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
