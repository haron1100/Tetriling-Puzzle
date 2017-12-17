import tensorflow as tf, numpy as np
import pickle
from utils import *
from copy import deepcopy

#random batching
def get_batch(size, data_size, x_train, y_train):
        x = np.array([])
        y = np.array([])
        for _ in range(size):
            k = np.random.choice(np.arange(data_size))
            x = np.append(x, x_train[k])
            
            y = np.append(y, y_train[k])
        x = x.reshape((size, 1, 7, 7))
        y = y.reshape((size, 76))
        return x, y

#linear batching
'''def get_batch(size, data_size, x_train, y_train, data_count):
        k = data_count%data_size
        k = np.clip(k, 0, data_size-size)
        x = x_train[k:k+size]
        y = y_train[k:k+size]
        x = x.reshape((size, 1, 7, 7))
        y = y.reshape((size, -1))
        return x, y'''


def train_neural_network(n_epochs):
    #model parameters
    inputs_shape = (7, 7)
    n_classes = 76

    #training parameters
    batch_size = 128

    #model creation
    x = tf.placeholder('float', shape=(None, 1, 7, 7), name='input_x')
    y = tf.placeholder('float')

    net = x
    # First convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                            filters=49, kernel_size=3, strides=2,
                            padding='same', activation=tf.nn.relu)
    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                            filters=49, kernel_size=3, strides=2,
                            padding='same', activation=tf.nn.relu)
    #Pooling Layer
    net = tf.nn.pool(input=net, name='pool', pooling_type='MAX',
                            padding='SAME', window_shape=(2, 2))
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dropout(inputs=net, rate=0.4, training=True)
    net = tf.layers.dense(inputs=net, name='o', units=n_classes, activation=None)
    output = net
    print('Model created')
    print('Loading dataset...')

    #importing data
    train_data_size = 400000
    test_data_size = 52000
    with open('conv_training_data_2m.pickle', 'rb') as file:
        dataset = pickle.load(file)

    x_train = dataset[0][:train_data_size]
    y_train = dataset[1][:train_data_size]
    x_train = x_train.reshape(train_data_size, 1, 7, 7)
    y_train = y_train.reshape(train_data_size, 76)
    
    x_test = dataset[0][train_data_size:train_data_size+test_data_size]
    y_test = dataset[1][train_data_size:train_data_size+test_data_size]
    x_test = x_test.reshape(test_data_size, 1, 7, 7)
    y_test = y_test.reshape(test_data_size, 76)
    
    print('Dataset loaded')
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005).minimize(cost)
    print('Training...', end='\n\n')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logfile = open('training_log', 'w')
        for epoch in range(n_epochs):
            epoch_loss=[]
            for _ in range(int(train_data_size/batch_size)):
                x1, y1 = get_batch(batch_size, train_data_size, x_train, y_train)
                _, c = sess.run([optimizer, cost], feed_dict={x:x1, y:y1})
                epoch_loss.append(c)
                
            print('Epoch', epoch, ' completed. Mean loss = ', np.mean(epoch_loss))
            logfile.write('Epoch ' + str(epoch) + '  Mean Loss: ' + str(np.mean(epoch_loss)) + '\n')
        print('Training complete')
        logfile.write('Training complete \n')
        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc = accuracy.eval({x:x_test, y:y_test})
        logfile.write('Accuracy: ' + str(acc))
        print('Accuracy: ', acc)

        saver = tf.train.Saver()
        # Save variables 
        print('Saving variables')
        saver.save(sess, "trained_conv_net", global_step=4)
        logfile.close()                    

train_neural_network(3)

