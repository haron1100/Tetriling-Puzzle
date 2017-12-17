import numpy as np 
import matplotlib as mp
#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
from utils import *
from PIL import Image

#model creation
x = tf.placeholder('float', shape=(None, 1, 7, 7), name='input_x')
y = tf.placeholder('float')

n_classes=76
net = x
# First convolutional layer.
hidden_1 = tf.layers.conv2d(inputs=net, name='layer_conv1',
                        filters=49, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu)
# Second convolutional layer.
hidden_2 = tf.layers.conv2d(inputs=hidden_1, name='layer_conv2',
                        filters=49, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu)

#Pooling Layer
pool_1 = tf.nn.pool(input=hidden_2, name='pool', pooling_type='MAX',
                        padding='SAME', window_shape=(2, 2))
net = tf.contrib.layers.flatten(pool_1)
net = tf.layers.dropout(inputs=net, rate=0.4, training=False)
output = tf.layers.dense(inputs=net, name='o', units=n_classes, activation=None)

#the tensorflow session is created and
#the trained model variables are restored from file
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "trained_conv_net-4")
print("Model restored.")

def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:stimuli})
    plotNNFilter(units)

'''def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(100,200))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], cmap="gray")'''
def plotNNFilter(units):
    units = np.clip(units, 0, 1)
    im  = units
    im = np.reshape(im, (units.shape[-2], units.shape[-1]))
    im = Image.fromarray(im*255)
    im.show()
    #print(np.shape(units))
    #print(units)
    filters = units.shape[3]
    for x in range(filters):
        im = units[:,:,:,x]
        im = np.reshape(im, (units.shape[-2], units.shape[-1]))
        im = Image.fromarray(im)
        im.show()
    
target = np.array(generate_target(7, 7, 0.5)).reshape((7, 7))
#target = np.zeros((7, 7))
#target[3:5, 3:5] = 1
#target[3, 3] = 1
#target[5, 3] = 1
#target[4, 3] = 1
#target[4, 4] = 1
im = target*255
im = Image.fromarray(im.astype('uint8'))
#im.show()
getActivations(hidden_1,target.reshape(-1, 1, 7, 7))


