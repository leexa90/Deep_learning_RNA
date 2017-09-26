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
data1 = np.load('../data_lim5000_nan_excludeTest.npy.zip')['data_lim5000_nan_excludeTest.npy'].item() #contains 480 train examples
data2 = np.load('../data_lim5000_ss_excludeTest.npy.zip')['data_lim5000_ss_excludeTest.npy'].item() #might have 494
data3 = np.load('../data_lim5000_extra_excludeTest.npy').item() #might have 494
data4 = np.load('../data_lim5000_MSA_excludeTest.npy').item() #might have 494

data1_t = np.load('../data_lim5000_nan.npy.zip')['data_lim5000_nan.npy'].item()
data2_t = np.load('../data_lim5000_ss.npy.zip')['data_lim5000_ss'].item()
data3_t = np.load('../data_lim5000_extra.npy').item()
data4_t = np.load('../data_lim5000_MSA.npy').item()

puzzle = ['3OX0', '3OWZ', '3OXJ', '3OXE', '3OWZ', '3OWW', '3OXM', '3OWW', '3OWI', '3OXE',
          '3OXM', '3OX0', '3OXJ', '3OWI', '3OXB', '3OXD', '3OXD', '3OXB', '4LCK', '4TZV',
          '4TZZ', '4TZZ', '4TZZ', '4LCK', '4TZZ', '4TZP', '4TZW', '4TZP', '4TZZ', '4TZZ',
          '4TZV', '4TZW', '4TZZ', '4LCK', '4TZZ', '4TZP', '4LCK', '4TZP', '5EAQ', '5DQK',
          '5EAO', '5DH6', '5DI2', '5DH8', '5DH7', '5DI4', '4R4V', '5V3I', '4R4P', '3V7E',
          '3V7E', '4L81', '4OQU', '4P9R', '4P95', '4QLM', '4QLN', '4XWF', '4XW7', '4GXY',
          '5DDO', '5DDO', '5TPY']

data_test = {}
data_train = {}
data_val = {}
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
data1_keys = [x for x in data1 if  len(data1[x][0]) <= 500] # the short ones get val-train split
random.shuffle(data1_keys)
data1_keys_train = data1_keys[0::3] + data1_keys[1::3] + [x for x in data1 if  len(data1[x][0]) > 500]
data1_keys_val = data1_keys[2::3]
data1_keys_test = [ x for x in data1_t if x[0:4].upper() in puzzle]
for i in data1_keys_train:
    if len(data1[i][0]) >= 35:
        if i in data2.keys():
            try:
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
                #data[i] = [tempF, temp1[0],temp1[1],temp_resi_map,d,temp2[1],temp2[0]]
                for window_tup in [(35,3),(50,3),(75,3),(100,3),(150,3),(200,3),(300,3),(400,3),(800,3)]:
                    window, jump = window_tup[0], window_tup[1]
                    for repeat in range(0,len(data1[i][0]) - window+1,jump):
                        if np.mean(d[repeat:repeat+window,repeat:repeat+window,:]) > 0.9: 
                            data_train[i+'_'+str(window)+'_'+str(repeat)] = [tempF[:,repeat:repeat+window],
                                                   temp1[0][repeat:repeat+window],
                                                   temp1[1][repeat:repeat+window],
                                                   temp_resi_map[repeat:repeat+window,repeat:repeat+window,:],
                                                   d[repeat:repeat+window,repeat:repeat+window,:],
                                                   temp2[1][repeat:repeat+window],
                                                   temp2[0][repeat:repeat+window,repeat:repeat+window]]
            except ValueError:
                print ('%s had ValueError, probably some of inputs are of wrong dimention. Data thrown away ' %i)
train_n = len(data_train)
for i in data1_keys_val:
    if len(data1[i][0]) >= 35:
        if i in data2.keys():
            try:
                temp1 = data1[i]
                a,b,c = (data1[i][2] < 8)*1,(data1[i][2] <= 15) & (data1[i][2] >= 8)*1,(data1[i][2] > 15)*1
                temp_resi_map = np.stack((a,b,c),axis=2)
                d = -1*(np.isnan(data1_t[i][2])-1) #non-nan values ==1 , nan =0
                d = np.stack((d,d,d),axis=2)
                
                temp2 = data2[i]
                temp3 = data3[i]
                temp4 = data4[i]
                tempF = np.concatenate((np.array(make_array(temp1[1])).T,np.array([temp2[1]]),temp3,(np.array(make_array(temp4[1])).T),np.array([make_array2(temp4[2])])))
                    #         [9-features, seq, exxist_seq, cat dist_map,cat dist_map (non-zero), ss_1d, ss_2d]
                data_val[i] = [tempF, temp1[0],temp1[1],temp_resi_map,d,temp2[1],temp2[0]]
            except ValueError:
                print ('%s had ValueError, probably some of inputs are of wrong dimention. Data thrown away ' %i)
for i in data1_keys_test:
    if len(data1_t[i][0]) >= 35:
        if i in data2_t.keys():
            try:
                temp1 = data1_t[i]
                a,b,c = (data1_t[i][2] < 8)*1,(data1_t[i][2] <= 15) & (data1_t[i][2] >= 8)*1,(data1_t[i][2] > 15)*1
                temp_resi_map = np.stack((a,b,c),axis=2)
                d = -1*(np.isnan(data1_t[i][2])-1) #non-nan values ==1 , nan =0
                d = np.stack((d,d,d),axis=2)
                
                temp2 = data2_t[i]
                temp3 = data3_t[i]
                temp4 = data4_t[i]
                tempF = np.concatenate((np.array(make_array(temp1[1])).T,np.array([temp2[1]]),temp3,(np.array(make_array(temp4[1])).T),np.array([make_array2(temp4[2])])))
                    #         [9-features, seq, exxist_seq, cat dist_map,cat dist_map (non-zero), ss_1d, ss_2d]
                data_test[i] = [tempF, temp1[0],temp1[1],temp_resi_map,d,temp2[1],temp2[0]]
            except ValueError:
                print ('%s had ValueError, probably some of inputs are of wrong dimention. Data thrown away ' %i)

                
val_n = len(data_val) 
print ('training samples %s from %s PDBid, val samples %s from %s PDBid' %(train_n,len(data1_keys_train),val_n,len(data1_keys_val)))            
#np.save('data_all.npy',data)
dictt = {}
length = []
for i in data_train.keys():
	if tuple(i.split('_')[0:2]) not in dictt :
		dictt[tuple(i.split('_')[0:2])] = 1
	else :
		dictt[tuple(i.split('_')[0:2])] += 1
	length  += [len(data_train[i][1]),]
#plt.hist([dictt[x] for x in dictt],100);plt.show()
#plt.hist(length,100);plt.show()
# initialzie train data here
data2_x = []
data2_y = []
data2_y_nan = []
data2_y_ss = []
data2_name = []
classweight1 = 0.
classweight2 = 0.
classweight3 = 0.
for i in data_train:
        data2_x += [data_train[i][0],]
        data2_y += [data_train[i][3],]
        data2_y_nan += [data_train[i][-3],]
        data2_y_ss += [data_train[i][-1],]
        data2_name += [i,]
        classweight1 += np.sum(data2_y[0][:,:,0] ==1)
        classweight2 += np.sum(data2_y[0][:,:,1] ==1)
        classweight3 += np.sum(data2_y[0][:,:,2] ==1)
weight1 = (classweight1 / (classweight1+classweight2+classweight3))**-1
weight2 = (classweight2 / (classweight1+classweight2+classweight3))**-1
weight3 = (classweight3 / (classweight1+classweight2+classweight3))**-1
weight_coeff1 = weight1 / (weight1+weight2+weight3) 
weight_coeff2 = weight2 / (weight1+weight2+weight3) 
weight_coeff3 = weight3 / (weight1+weight2+weight3)
print ('Class 1 has %s percent, given %s weight' % (weight1**-1, weight_coeff1))
print ('Class 2 has %s percent, given %s weight' % (weight2**-1, weight_coeff2))
print ('Class 3 has %s percent, given %s weight' % (weight3**-1, weight_coeff3))
# making a batch 
data3_x = {}
data3_y = {}
data3_y_nan = {}
data3_y_ss = {}
data3_name = {}
for i,j in [(35,1),(50,1),(75,1),(100,1),(150,1),(200,1),(300,1),(400,1),(800,1)]:
    data3_x[i] = []
    data3_y[i] = []
    data3_y_nan[i] = []
    data3_y_ss[i] = []
    data3_name[i] = []
    for k in [l for l in range(len(data2_name)) if int(data2_name[l].split('_')[2] ) ==i]:
            data3_x[i] += [data2_x[k],]
            data3_y[i] += [data2_y[k],]
            data3_y_nan[i] += [data2_y_nan[k],]
            data3_y_ss[i] += [data2_y_ss[k],]
            data3_name[i] += [data2_name[k],]
# initialzie val data here
data2_x_val = []
data2_y_val = []
data2_y_nan_val = []
data2_y_ss_val = []
data2_name_val = []       
for i in data_val:
        data2_x_val += [data_val[i][0],]
        data2_y_val += [data_val[i][3],]
        data2_y_nan_val += [data_val[i][-3],]
        data2_y_ss_val += [data_val[i][-1],]
        data2_name_val += [i,]
print (len(data2_y),'finished intitialising total number of training/val samples')
# initialzie test data here
data2_x_test = []
data2_y_test = []
data2_y_nan_test = []
data2_y_ss_test = []
data2_name_test = []       
for i in data_test:
        data2_x_test += [data_test[i][0],]
        data2_y_test += [data_test[i][3],]
        data2_y_nan_test += [data_test[i][-3],]
        data2_y_ss_test += [data_test[i][-1],]
        data2_name_test += [i,]
#data2_y = np.array(data2_y)
#data2_x = np.array(data2_x)
# Create some wrappers for simplicity
print  (data2_name_val  )
epsilon = 1e-3
def batch_normalization(x):
    mean,var = tf.nn.moments(x,[0,1,2])
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
learning_rate = 0.0001
training_epochs = 20 
batch_size = 1
display_step = 1
n_classes =5



# tf Graph input
x = tf.placeholder(tf.float32,[None,15,None,1])
# pairwise distances need to define
resi_map0 = tf.placeholder(tf.float32,[None,None,None,3])
# some values are na, these will be excluded from loss
above_zero = tf.placeholder(tf.float32,[None,None,None,3])
above_zero = tf.cast(above_zero,dtype=tf.float32) #TF to float
# if tf.is_nan, use convert to 0, else use original values
resi_map = tf.where(tf.is_nan(resi_map0),above_zero,resi_map0)
# ss_2d
ss_2d = tf.placeholder(tf.float32,[None,None,None])
#resi_map = tf.reshape(resi_map, shape=[1, -1, -1, 1])
#x = tf.reshape(x, shape=[1, -1, 4, 1])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
window = 15
num1 = 32/4
num2 = 64/4
num3 = 128/4
num4 = 64/4
num5 = 64/4
num6 = 64/4
num7 = 64/4
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
    '3_wc1bb': tf.Variable(tf.random_normal([4, 4, num3, num3])),
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
    '4_wc1bb': tf.Variable(tf.random_normal([4, 4, num4, num4])),
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

    '9_out2': tf.Variable(tf.random_normal([5,5,num6*4, 3]))   
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
conv2da = conv2d(conv1,weights['2_wc1d'],biases['2_bc1d']) #not to over helper function with name, changed variable from conv2d to conv2da
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

out = conv2d(conv7,weights['9_out2'],biases['9_out2'],relu=False)
out_softmax = tf.nn.softmax(out,-1,name='softmax')
#out_norm = batch_normalization(out)
# kill entries of nan so they are not in cost, not needed ???
#out_softmax2 = tf.multiply(out_softmax,above_zero,name='zero') 


# Define loss and optimizer
logit_weight = tf.constant([weight_coeff1,weight_coeff2,weight_coeff3],tf.float32)
log_loss =tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = tf.multiply(resi_map,logit_weight) ,dim=-1)
#log_loss = -tf.reduce_sum(tf.multiply(resi_map ,tf.log(out_softmax+10**-15)),-1)
cost = tf.reduce_mean(log_loss)
#cost = tf.div(sum_logLoss,tf.reduce_sum(above_zero))
optimizer = tf.train.AdamOptimizer(epsilon = 0.0001,learning_rate=learning_rate).minimize(cost)
def accuracy(mat_model,answer):
    score = [[],[],[]]
    for i in range(0,answer.shape[1]):
        if np.sum(answer[i,:]) != 0:
            for j in range(i+1,answer.shape[1]):
                if np.sum(answer[j,:]) != 0:
                    if answer[i,j] == mat_model[i,j]:
                        score[answer[i,j]] += [1,]
                    else:
                        score[answer[i,j]] += [0,]
    return np.mean([np.mean(score[0]),np.mean(score[1]),np.mean(score[2])])

saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver.restore(sess,'./model300_reweigh_loss.ckpt')

# Training cycle
result = {}
random.seed(0)
shuffle = range(len(data2_x))
random.shuffle(shuffle)
text = ''
for epoch in range(training_epochs):
    counter = 0
    avg_cost = []
    val_cost = []
    test_cost = []
    train_acc = []
    val_acc = []
    test_acc = []
    total_batch = train_n#int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for size in sorted(data3_x.keys())[0:0]:
        shuffle = range(len(data3_x[size]))
        random.shuffle(shuffle)
        if size >=400:
            num = 1
        else:
            num = 8
        for batch in range(0,len(shuffle),num):
            batch_list = shuffle[batch:batch+num] 
            counter += 1
            #print (i)
            batch_x = np.array([[data3_x[size][i]] for i in batch_list])
            batch_y = np.array([data3_y[size][i]for i in batch_list])
            batch_y_nan = np.array([data3_y_nan[size][i]  for i in batch_list])
            batch_y_ss = np.array([data3_y_ss[size][i]  for i in batch_list ])
            batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
            # Run optimization op (backprop) and cost op (to get loss value)
            #_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,resi_map0: batch_y,above_zero : batch_y_nan,ss_2d : batch_y_ss})
            pred = sess.run( out_softmax, feed_dict={x: batch_x,resi_map0: batch_y,above_zero : batch_y_nan, ss_2d : batch_y_ss})
            for k in range(len(batch_y)):
                train_acc += [accuracy(np.argmax(pred[k],3)[0],np.argmax(batch_y[k],3)[0]),]
            # Compute average loss
            avg_cost += [c,]
    if True:
        val_acc = []
        for i in range(len(data2_x_val)):
                batch_x, batch_y = np.array([[data2_x_val[i],],]),np.array([data2_y_val[i],])
                batch_y_nan,batch_y_ss = np.array([data2_y_nan_val[i]]),np.array([data2_y_ss_val[i]])
                batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
                cost_i  = sess.run( cost, feed_dict={x: batch_x,resi_map0: batch_y,
                                                     above_zero : batch_y_nan, ss_2d : batch_y_ss})
                val_cost += [cost_i,]
                pred =sess.run( out_softmax, feed_dict={x: batch_x,resi_map0: batch_y,
                                                     above_zero : batch_y_nan, ss_2d : batch_y_ss})
                val_acc += [accuracy(np.argmax(pred+np.transpose(pred,(0,2,1,3)),3)[0],np.argmax(batch_y,3)[0]),]
                print (data2_name_val[i],val_acc[-1],batch_y.shape)
        test_acc = []
        for i in range(len(data2_x_test))[-4:-3]:
                batch_x, batch_y = np.array([[data2_x_test[i],],]),np.array([data2_y_test[i],])
                batch_y_nan,batch_y_ss = np.array([data2_y_nan_test[i]]),np.array([data2_y_ss_test[i]])
                batch_x = np.swapaxes(np.swapaxes(batch_x,1,3),1,2)
                cost_i  = sess.run( cost, feed_dict={x: batch_x,resi_map0: batch_y,
                                                     above_zero : batch_y_nan, ss_2d : batch_y_ss})
                test_cost += [cost_i,]
                pred =sess.run( out_softmax, feed_dict={x: batch_x,resi_map0: batch_y,
                                                     above_zero : batch_y_nan, ss_2d : batch_y_ss})
                test_acc += [accuracy(np.argmax(pred+np.transpose(pred,(0,2,1,3)),3)[0],np.argmax(batch_y,3)[0]),]
                print (data2_name_test[i],test_acc[-1],batch_y.shape)
        print (np.mean(test_acc))
    # Display logs per epoch step
    #f1 = open('updates.log','w')
    text += str(np.mean(avg_cost))+'  '+str(np.mean(train_acc))+'\n'
    text += str(np.mean(val_cost))+'  '+str(np.mean(val_acc))+'\n\n'
    print ("Epoch:", '%04d' % (epoch+1), "cost=", 
            "{:.9f}".format(np.mean(avg_cost)),np.mean(train_acc))
    print ("Epoch:", '%04d' % (epoch+1), "cost=", 
            "{:.9f}".format(np.mean(val_cost)),np.mean(val_acc))
    #f1.write(text)
    #f1.close()
    #save_path = saver.save(sess,'model300_reweigh_loss.ckpt')
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

plt.plot(range(0,training_epochs),[result[i][0] for i in result],label='Train')
plt.plot(range(0,training_epochs),[result[i][1] for i in result],label='Val')
plt.legend();plt.ylabel('Logloss cost') ; plt.xlabel('epoch')
plt.savefig('Train_curveLg.png')

