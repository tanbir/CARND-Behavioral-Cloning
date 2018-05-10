# Author: Tanbir Ahmed

# Files to import
###########################################################################################
# Modeling using keras and scikit-learn
import keras
from keras import __version__ as keras_version
print('Keras version: ', keras_version)
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Images and plotting
from PIL import Image         
import cv2
import matplotlib.pyplot as plt

# Miscellaneous
import numpy as np
import math
import csv
import os
###########################################################################################

###########################################################################################
# Configuration

config = {
    'data_path_udacity' : './udacity_data',
    'data_path' : './training_data'
}
###########################################################################################

###########################################################################################
# Data loading class
class load_data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.driving_data = []
        self.image_paths = []
        self.angles = []
        
    def read_csv(self):
        with open(self.data_path+'/driving_log.csv', newline='') as f:
            self.driving_data = list(
                csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE)
            )
        
    def extract_image_info(self):
        self.read_csv()
        for row in self.driving_data[1:]:
            speed = row[6]
            
            self.image_paths.append(self.data_path + '/IMG/' + os.path.basename(row[0]))
            self.angles.append(float(row[3]))
            
            self.image_paths.append(self.data_path + '/IMG/' + os.path.basename(row[1]))
            self.angles.append(float(row[3])-0.25)
            
            self.image_paths.append(self.data_path + '/IMG/' + os.path.basename(row[2]))
            self.angles.append(float(row[3])+0.25)
    
    def get_images_and_angles(self):
        self.extract_image_info()
        self.image_paths, self.angles = shuffle(self.image_paths, self.angles)
        images = []
        angles = []
        for i in range(len(self.angles)):
            img = cv2.imread(self.image_paths[i])
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize images
            img = cv2.resize(img,(200, 100), interpolation = cv2.INTER_AREA)
            angle = self.angles[i]
            images.append(img)
            angles.append(angle)
            # Add flipped images based on minimum value of the angle
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                images.append(img)
                angles.append(angle)
        return shuffle(np.array(images), np.array(angles))
###########################################################################################

###########################################################################################
# Model design class
class RegressorModel():
    def __init__(self):
        self.model = None
        
    def add_conv(self, filters, kernel_size, strides=1, border='valid', reg=0.001):
        self.model.add(Convolution2D(filters, kernel_size, kernel_size, 
                                     subsample=(strides, strides), 
                                     border_mode=border, W_regularizer=l2(reg), 
                                     activation='relu')
                      )
        
    def add_fc(self, size, reg=0.001):
        self.model.add(Dense(size, W_regularizer=l2(0.001), activation='relu'))
        
    # Using the nVidia Autonomous car group model
    def build(self, input_shape):        
        self.model = Sequential()
        
        # Normalize using Lambda layer used to create arbitrary functions 
        # on each image as it passes through the layer
        self.model.add(Lambda(lambda x: x/255 - 0.5, input_shape=input_shape))
        
        # Cropping inside the model is quite fast since the model is parallelized on the GPU
        #self.model.add(Cropping2D(cropping=((50,20), (0,0))))
        
        print(self.model.output_shape)

        self.add_conv(24, 5, 2)
        self.add_conv(36, 5, 2)
        self.add_conv(48, 5, 2)
        self.add_conv(64, 3, 1)
        self.add_conv(64, 3, 1)        
        
        self.model.add(Flatten())
        
        self.add_fc(100)
        self.add_fc(50)
        self.add_fc(10)
        
        self.model.add(Dense(1))

    
    def compile(self, rate):
        opt = keras.optimizers.Adam(lr=rate)
        self.model.compile(loss='mse', 
                           optimizer=opt, 
                           metrics=['accuracy'])
        
    def fit(self, file, nb_epoch):
        checkpointer = ModelCheckpoint(filepath=file, monitor='val_loss', verbose=1, save_best_only=True)

        if os.path.isfile(file):
            self.model.load_weights(file)
        self.history_object = self.model.fit(x_train, y_train, batch_size=128, nb_epoch=nb_epoch, 
                                             validation_split=0.2, callbacks=[checkpointer], verbose=2, shuffle=True)  
        
    def summary(self):
        self.model.summary()
        
    # print the keys contained in the history object
    def visualize(self):
        print(self.history_object.history.keys())

        ### plot the training and validation loss for each epoch
        plt.plot(self.history_object.history['loss'])
        plt.plot(self.history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
###########################################################################################


print('Loading data')
l = load_data(config['data_path'])
x_train = None
y_train = None
x_train, y_train = l.get_images_and_angles()
print('Data loaded', x_train.shape, y_train.shape)

print('Bulding and compiling model')
m = RegressorModel()
m.build(input_shape=(x_train.shape[1:]))
m.summary()

m.compile(rate=1e-4)

print('Training...')
m.fit('model.h5', nb_epoch=10)

print('How did the training go?')
m.visualize()