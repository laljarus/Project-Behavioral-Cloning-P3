# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:58:26 2017

@author: laljarus
"""

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
#import sklearn

  

filename = ".\Test\driving_log.csv"
FolderPath = ".\\Test\\IMG\\"
lines=[]
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

arrImages = []
arrSteerAng = []
correctionFactor = 0.2
    
for line in lines:
    # Center Image
    SourcePath_center = line[0]
    filename_center = SourcePath_center.split("\\")[-1]
    CurrentPath_center = FolderPath + filename_center
    Image_center = cv2.imread(CurrentPath_center)
    arrImages.append(Image_center)        
    SteerAng_center = float(line[7])           
    arrSteerAng.append(SteerAng_center)
    
    # Center Flip Image        
    arrImages.append(cv2.flip(Image_center,1))
    arrSteerAng.append(-1*SteerAng_center)
    
    # Left Image
    SourcePath_left = line[1]
    filename_left = SourcePath_left.split("\\")[-1]
    CurrentPath_left = FolderPath + filename_left
    Image_left = cv2.imread(CurrentPath_left)
    arrImages.append(Image_left)        
    SteerAng_left = SteerAng_center + correctionFactor
    arrSteerAng.append(SteerAng_left)
           
    # Left Image
    SourcePath_right = line[2]
    filename_right = SourcePath_right.split("\\")[-1]
    CurrentPath_right = FolderPath + filename_right
    Image_right = cv2.imread(CurrentPath_right)
    arrImages.append(Image_right)        
    SteerAng_right = SteerAng_center - correctionFactor
    arrSteerAng.append(SteerAng_right)        
    
    
X_train = np.array(arrImages)
y_train = np.array(arrSteerAng)
    
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
model.add(Dropout(0.7))
model.add(Dense(1))
model.add(Activation('tanh'))

#sgd = optimizers.SGD(lr=0.01)
model.compile(loss = "mse",optimizer = 'adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,validation_split = 0.2,shuffle = True\
                    ,epochs = 4,batch_size = 32)
					

model.save('model.h5')
    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()