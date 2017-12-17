# ####################################################
# DE2-COM2 Computing 2
# Individual project
#
# Title: MAIN FILE EXAMPLE
# Authors: Liuqing Chen, Feng Shi, 
#          and Isaac Engel (13th September 2017)
# Last updated: 13th September 2017
# ####################################################
import tensorflow as tf, numpy as np
import pickle
from utils import *

'''
Algorithm works by scanning the target starting from the top left corner and
going to the bottom right stopping each time it encounters a one and classifying
it into one of 76 squares belonging to one of the 19 shapes before adding the
shape to the solution and removing it from the target. The classification
is done by looking at the 7x7 grid around the square and eliminating all the
shapes which do not fit into the that area. It does this by looking at the
neighbouring squares around the square being classified. 
'''

#filters is a list of coordinates relative to the square being classified
#that should be checked to classify the square
num_filters = 24
filters = [(0, 0) for _ in range(num_filters)]

#first filter - one above
filters[0] = (0, 1)

#second filter - one below
filters[1] = (0, -1)

#third filter - one to the right
filters[2] = (1, 0)

#fourth filter - one left
filters[3] = (-1, 0)

#fifth filter - two above
filters[4] = (0, 2)

#sixth filter - two to the right
filters[5] = (2, 0)

#seventh filter - top right diagonal
filters[6] = (1, 1)

#eighth filter - top left diagonal
filters[7] = (-1, 1)

#ninth filter - bottom right diagonal
filters[8] = (1, -1)

#tenth filter - bottom left diagonal
filters[9] = (-1, -1)

#eleventh filter - two to the left
filters[10] = (-2, 0)

#twelvth filter - two below
filters[11] = (0, -2)

#thirteenth filter - two above and one right
filters[12] = (1, 2)

#fourteenth filter - two above and one left
filters[13] = (-1, 2)

#fifteenth filter - two below and one right
filters[14] = (1, -2)

#sixteenth filter - two below and one left
filters[15] = (-1, -2)

#seventeenth filter - two left and one above
filters[16] = (-2, 1)

#eighteenth filter - two left and one below
filters[17] = (-2, -1)

#nineteenth filter - two right and one above
filters[18] = (2, 1)

#20th filter - two right and one below
filters[19] = (2, -1)

#21st filter - three above
filters[20] = (0, 3)

#22nd filter - three to the right
filters[21] = (3, 0)

#23rd filter - three below
filters[22] = (0, -3)

#24th filter - three to the left
filters[23] = (-3, 0)


#the neural network model is created below in tensorflow
#model parameters
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
net = tf.layers.dropout(inputs=net, rate=0.4, training=False)
net = tf.layers.dense(inputs=net, name='o', units=n_classes, activation=None)
output = net

#the tensorflow session is created and
#the trained model variables are restored from file
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "trained_conv_net-3")
#print("Model restored.")

# Write or import your functions in this file
def Tetris(target):
    

    #creates padding around the target to prevent the 'out of bounds' error
    #when checking neighbouring area around the square
    target_shape = np.shape(target)
    nrows = target_shape[0]
    ncols = target_shape[1]
    target = np.pad(target, 3, 'constant', constant_values=0)

    print(target, end='\n\n')
    
    #noshapes and solution array initialization
    noshapes = 0
    solution = [[(0, 0)]*(ncols) for x in range(nrows)]
    print('Solution: ', solution)

    #start scanning for squares which need to be filled starting from the top
    #left to the bottom right
    for i in range(3, np.shape(target)[0]-4):
            for j in range(3, np.shape(target)[1]-4):
                    #if the square needs to be filled and there is not a previously
                    #placed shape on the square
                    if target[i][j] != 0 and solution[i-3][j-3]== (0, 0):
                        #get input of 7x7 grid around square
                        #for classification algorithm
                        submat = target[i-3: i+4, j-3:j+4].reshape((7, 7))
                        print('Submat: ', submat)
                        #each block id is assumed to fit until eliminated
                        possible_class = np.arange(76)
                        #for each filter checks the required square around one being classified
                        #to eliminate ones that do not fit
                        for k in range(np.shape(filters)[0]):
                            fits = True
                            if submat[3 - filters[k][1]][3 + filters[k][0]] == 0 or solution[i-3 - filters[k][1]][j-3 + filters[k][0]]!=(0, 0):
                                fits = False
                            #is there a block above
                            if k==0:
                                if not fits:
                                    possible_class[[2, 3, 5, 6, 7, 13, 14, 19, 22, 23, 27, 29, 31, 35, 38, 39, 41, 45, 47, 50, 54, 55, 59, 63, 65, 67, 70, 74, 75]] = -1
                            #is there a block below
                            elif k==1:
                                if not fits:
                                    possible_class[[0, 1, 4, 5, 6, 12, 13, 16, 21, 22, 24, 28, 29, 34, 36, 38, 40, 44, 45, 48, 52, 54, 57, 60, 64, 66, 69, 72, 73]] = -1
                            #is there a block to the right
                            elif k==2:
                                if not fits:
                                    possible_class[[0, 2, 8, 9, 10, 14, 17, 18, 20, 24, 25, 30, 32, 33, 36, 41, 42, 45, 49, 50, 53, 56, 57, 60, 62, 65, 68, 70, 73]] = -1
                            #is there a block to the left
                            elif k==3:
                                if not fits:
                                    possible_class[[1, 3, 9, 10, 11, 15, 18, 19, 21, 25, 26, 31, 33, 34, 37, 42, 43, 46, 50, 51, 54, 57, 58, 61, 63, 66, 69, 71, 74]] = -1
                            #two above
                            elif k==4:
                                if not fits:
                                    possible_class[[6, 7, 14, 23, 31, 39, 47, 55]] = -1
                            #two to the right
                            elif k==5:
                                if not fits:
                                    possible_class[[8, 9, 17, 24, 32, 41, 49, 56]] = -1
                            #top right diagonal
                            elif k==6:
                                if not fits:
                                    possible_class[[2, 18, 27, 30, 38, 47, 49, 53, 59, 62, 63, 73, 75]] = -1
                            #top left diagonal
                            elif k==7:
                                if not fits:
                                    possible_class[[3, 15, 22, 35, 42, 46, 51, 55, 59, 66, 67, 70, 71]] = -1
                            #bottom right diagonal
                            elif k==8:
                                if not fits:
                                    possible_class[[0, 13, 20, 33, 40, 44, 48, 53, 56, 64, 65, 68, 69]] = -1
                            #bottom left diagonal
                            elif k==9:
                                if not fits:
                                    possible_class[[1, 16, 25, 29, 37, 46, 48, 52, 58, 60, 61, 72, 74]] = -1
                            #two to the left
                            elif k==10:
                                if not fits:
                                    possible_class[[10, 11, 19, 26, 34, 43, 51, 58]] = -1                
                            #two below
                            elif k==11:
                                if not fits:
                                    possible_class[[4, 5, 12, 21, 28, 36, 44, 52]] = -1
                            #two above and one right
                            elif k==12:
                                if not fits:
                                    possible_class[[30, 39, 75]] = -1
                            #two above and one left             
                            elif k==13:
                                if not fits:
                                    possible_class[[15, 23, 67]] = -1
                            #two below one right             
                            elif k==14:
                                if not fits:
                                    possible_class[[12, 20, 64]] = -1
                            #two below one left   
                            elif k==15:
                                if not fits:
                                    possible_class[[28, 37, 72]] = -1
                            #two left one above
                            elif k==16:
                                if not fits:
                                    possible_class[[35, 43, 71]] = -1
                            #two left one below    
                            elif k==17:
                                if not fits:                
                                    possible_class[[16, 26, 61]] = -1
                            #two right and one above             
                            elif k==18:
                                if not fits:
                                    possible_class[[17, 27, 62]] = -1
                            #two right and one below
                            elif k==19:
                                if not fits:
                                    possible_class[[32, 40, 68]] = -1
                            #three above
                            elif k==20:
                                if not fits:
                                    possible_class[[7]] = -1
                            #three to the right
                            elif k==21:
                                if not fits:
                                    possible_class[[8]] = -1                
                            #three below
                            elif k==22:
                                if not fits:
                                    possible_class[[4]] = -1 
                            #three to the left
                            elif k==23:
                                if not fits:
                                    possible_class[[11]] = -1 
                        #possible class is turned into a list of all the possible
                        #classifications that do not lead to an error
                        possible_class = [aclass for aclass in possible_class if aclass!=-1]
                        print('Possible class: ', possible_class)
                        #if there exists a shape that can fit
                        if np.shape(possible_class)[0]!=0:
                            if np.shape(possible_class)[0]==1:
                                o = possible_class[0]
                            else:
                                #classify it using the convolutional neural network
                                o = sess.run([output], feed_dict={x:submat.reshape(-1, 1, 7, 7)})
                                o = np.argmax(o)
                                print('NN output: ', o)
                                #if the guessed classification of the neural network leads to an error
                                if o not in possible_class:
                                    #choose the first shape that fits
                                    o = possible_class[0]
                                    print('NN wrong, ', o, ' chosen')

                            #place shape in solution and remove from target
                            blockid = o
                            noshapes+=1
                            shapeid = int(blockid/4)+1
                            shape = generate_shape(shapeid)
                            blockind = blockid%4
                            for block in shape:
                                solution[i-3+block[0]- shape[blockind][0]][j-3+block[1] - shape[blockind][1]] = (shapeid, noshapes)
                                target[i+block[0]- shape[blockind][0]][j+block[1] - shape[blockind][1]] = 0
                        print('Target: ', target)
                        print('Solution', solution, end='\n\n')
    
    return solution
