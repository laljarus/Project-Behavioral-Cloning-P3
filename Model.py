# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 00:14:38 2017

@author: laljarus
"""
# Importing the reqired libraries

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten,Dropout,Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

# Extracting the test and valiudation data from the simulator logs

filename = "C:\\Users\\laljarus\\Documents\\GitHub\\Test2\\driving_log.csv"
FolderPath = "C:\\Users\\laljarus\\Documents\\GitHub\\Test2\\IMG\\"
lines=[]

SteerAngRaw = []
FileNamesCenter = []
FileNamesLeft = []
FileNamesRight = []
correctionFactor = 0.2

with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:                
        FileNamesCenter.append(line[0].split("\\")[-1])        
        FileNamesLeft.append(line[1].split("\\")[-1])       
        FileNamesRight.append(line[1].split("\\")[-1])       
        SteerAngRaw.append(float(line[3]))

# Preprocessing of the steerubg angle value from the simulator
# Running Average Filter
# N is the filter order of mean average filter
# The running average value is unused
N = 5 
window_array = np.ones(N)/N
SteerAngFiltMean = np.convolve(SteerAngRaw,window_array,mode = 'same')


'''
arrImages = []
arrSteerAng = []

for FileNameCenter,FileNameRight,FileNameLeft,SteerAng in \
zip(FileNamesCenter,FileNamesLeft,FileNamesRight,SteerAngRaw):
    
    ImgCenter = cv2.imread(FolderPath+FileNameCenter)
    ImgCenter = cv2.cvtColor(ImgCenter , cv2.COLOR_BGR2RGB)
    arrImages.append(ImgCenter)
    arrSteerAng.append(SteerAng)
    
    ImgFlip = cv2.flip(ImgCenter,1)
    arrImages.append(ImgFlip)
    arrSteerAng.append(-1*SteerAng)
    
    ImgLeft = cv2.imread(FolderPath+FileNameLeft)
    ImgLeft = cv2.cvtColor(ImgLeft , cv2.COLOR_BGR2RGB)
    arrImages.append(ImgLeft)
    arrSteerAng.append(SteerAng+correctionFactor)
    
    ImgRight = cv2.imread(FolderPath+FileNameRight)
    ImgRight = cv2.cvtColor(ImgRight , cv2.COLOR_BGR2RGB)
    arrImages.append(ImgRight)
    arrSteerAng.append(SteerAng-correctionFactor)
    
X_train = np.array(arrImages)
y_train = np.array(arrSteerAng)
'''
# Storing the file names of images and steering angle in a dataframe 
# which is processed by generator

samples_dict = {'FileNameCenter':FileNamesCenter,'FileNameLeft':FileNamesLeft,\
                'FileNameRight':FileNamesRight,'SteerAng':SteerAngRaw}
samples_df = pd.DataFrame(samples_dict)

# Generator fetches the test and training data to the neural network

def Generator(samples,batch_size=32,correctionFactor = 0.2):
    #shuffle(samples)
    num_samples = len(samples)
    
    while 1:
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            arrImages = []
            arrSteerAng = []
            for ix,sample in batch_samples.iterrows():
                FileNameCenter = sample['FileNameCenter']
                FileNameLeft = sample['FileNameLeft']
                FileNameRight = sample['FileNameRight']
                SteerAng = sample['SteerAng']
                
                # Image from the center camera
                
                ImgCenter = cv2.imread(FolderPath+FileNameCenter)
                ImgCenter = cv2.cvtColor(ImgCenter , cv2.COLOR_BGR2RGB)
                arrImages.append(ImgCenter)
                arrSteerAng.append(SteerAng)
    
                # Flipped image from the center camera
                ImgFlip = cv2.flip(ImgCenter,1)
                arrImages.append(ImgFlip)
                arrSteerAng.append(-1*SteerAng)
                
                #Image from the left camera
                ImgLeft = cv2.imread(FolderPath+FileNameLeft)
                ImgLeft = cv2.cvtColor(ImgLeft , cv2.COLOR_BGR2RGB)
                arrImages.append(ImgLeft)
                arrSteerAng.append(SteerAng+correctionFactor)
                
                # Image from the right camera
                ImgRight = cv2.imread(FolderPath+FileNameRight)
                ImgRight = cv2.cvtColor(ImgRight , cv2.COLOR_BGR2RGB)
                arrImages.append(ImgRight)
                arrSteerAng.append(SteerAng-correctionFactor)                

            # trim image to only see section with road
            X_train = np.array(arrImages)
            y_train = np.array(arrSteerAng)
            yield shuffle(X_train, y_train)

# Spliting the test and training data 
train_samples,test_samples = train_test_split(samples_df)
# Generator to fetch the training data
train_generator = Generator(train_samples)
# Generator to fetch the test data
test_generator = Generator(test_samples)

# parameter to choose the network architecture
network = 'lenet' 

if network == 'nvidia':
   # Nvidia network architecture 
    model = Sequential()
    model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x:x/255-0.5))
    model.add(Convolution2D(24,(5, 5),activation = 'relu'))
    model.add(Convolution2D(36,(5, 5),activation = 'relu'))
    model.add(Convolution2D(48,(5, 5),activation = 'relu'))
    model.add(Convolution2D(64,(3, 3),activation = 'relu'))
    model.add(Convolution2D(64,(3, 3),activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('tanh'))
    model.add(Dense(100))
    model.add(Activation('tanh'))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dense(10))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    
elif network == 'lenet':
    
    # Lenet network architecture
    model = Sequential()
    model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x:x/255-0.5))
    model.add(Convolution2D(6,(5, 5),activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(16,(5, 5),activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dense(84))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    

batch_size = 32

#sgd = optimizers.SGD(lr=0.01)
#adm = optimizers.adam(lr = 0.001)
model.compile(loss = "mse",optimizer = 'adam')
history = model.fit_generator(train_generator, len(train_samples)/batch_size, \
                              epochs =  3 , validation_data = test_generator,\
                              validation_steps= len(test_samples)/batch_size)


model.save('model.h5')
    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.title('mean squared error loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.grid(True)
plt.show()
