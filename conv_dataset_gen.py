'''
Generates training data required to train a convolutional
neural network to classify a square based on the squares around it
'''
from utils import *
import pickle
from copy import deepcopy

class data_set:
    def __init__(self, target_shape=(30, 30), no_data_points=10000):
        self.generate_dataset(target_shape, no_data_points)
        self.no_data_points = no_data_points
        
    def generate_dataset(self, target_shape=(30, 30), no_data_points=10000):
        #initialize dataset
        self.x_train = np.zeros(no_data_points*49).reshape(-1, 7, 7)
        self.y_train = np.zeros(no_data_points*76).reshape(-1, 76)
        target_shape = (100, 100)
        
        #keeps generating data until data_count=no_data_points
        data_count = 0
        done = 0
        while not done:
            target, label = self.generate_target_labels(target_shape[1], target_shape[0], 0.6)
            target = np.pad(target, 3, 'constant', constant_values=0)
            label = np.pad(label, 3, 'constant', constant_values=0)
            for i in range(3, np.shape(target)[0]-3):
                            for j in range(3, np.shape(target)[1]-3):
                                    if target[i][j] != 0:
                                        
                                        #get 7x7 submatrix from target matrix
                                        submat = target[i-3: i+4, j-3:j+4].reshape((7, 7))
                        
                                        #append input and label to the dataset
                                        action = label[i][j]
                                        one_hot_encoding = np.zeros(76)
                                        one_hot_encoding[action] = 1

                                        self.x_train[data_count] = submat
                                        self.y_train[data_count] = one_hot_encoding                                        

                                        #update target with by removing 1s where block has been placed
                                        blockid = action
                                        shapeid = int(blockid/4)+1
                                        shape = generate_shape(shapeid)
                                        blockind = blockid%4
                                        for block in shape:
                                            target[i+block[0]- shape[blockind][0]][j+block[1] - shape[blockind][1]] = 0
                                        data_count+=1


                                        if data_count%50000 == 0:
                                            print(data_count, ' data points created out of ', no_data_points)                                        
                                    if data_count>=no_data_points:
                                        done=1
                                        break
                            if done:
                                break
            
        return self.x_train, self.y_train

    #Generates target-solution pair
    #The solution is given in the form of the block id (ranging from 0-75) for each square
    def generate_target_labels(self, width, height, density):
        """
        Generates a random solvable target shape
        NOTE: this function may not be able to generate targets with density above 0.8, so it is
        recommended to keep it below that value.
        :param width: number of columns of the target (must be positive)
        :param height: number of rows of the target (must be positive)
        :param density: number of columns of the target (must be between 0 and 1, recommended < 0.8)
        """
        assert width > 0, "width must be a positive integer"
        assert height > 0, "height must be a positive integer"
        assert 0 <= density <= 1, "density must be a number between 0 and 1"
        size = width * height
        nblocks = size * density
        npieces, _ = divmod(nblocks, 4)
        npieces = int(npieces)
        target = [[0] * width for row in range(0, height)]
        labels = deepcopy(target)
        for count in range(0, npieces):
            valid_piece = False
            end_counter = 0
            while not valid_piece and end_counter < 1000:
                r = int(random.uniform(0, height))
                c = int(random.uniform(0, width))
                shape_id = int(random.uniform(0, 18)) + 1
                shape = generate_shape(shape_id)
                piece = [[y + r, x + c] for [y, x] in shape]
                valid_piece = check_if_piece_is_valid(piece, target)
                if valid_piece:
                    for [r, c] in piece: 
                        target[r][c] = 1
                    for block_no in range(len(piece)):
                        block_id = ((shape_id-1)*4) + block_no
                        labels[piece[block_no][0]][piece[block_no][1]] = block_id
                        
                end_counter += 1
        return target, labels

    def save_dataset(self, fname):
        datafile = [self.x_train, self.y_train]
        with open(fname, 'wb') as file:
            pickle.dump(datafile, file)

print('Generating dataset')
mydat = data_set(target_shape=(100,100), no_data_points=452000)
print('Saving dataset')
mydat.save_dataset('conv_training_data_2m.pickle')

