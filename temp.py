import tensorflow as tf

from keras import backend as K
sess = K.get_session()
K.set_session(sess)
z = tf.Variable([[1,1],[0,4],[5,6]])
K.eval(z)
die


