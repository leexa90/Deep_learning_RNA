import tensorflow as tf

from keras import backend as K
import numpy as np
data = np.load('data_0nan.npy.zip')['data_0nan'].item()

sess = K.get_session()
K.set_session(sess)
z = tf.Variable([[1,1],[0,4],[5,6]])
K.eval(z)
die

