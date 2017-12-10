# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 00:14:38 2017

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
#from scipy.fftpack import fft
from scipy import signal
#import sklearn
filename = "C:\\Users\\laljarus\\Documents\\GitHub\\Test2\\driving_log.csv"
FolderPath = "C:\\Users\\laljarus\\Documents\\GitHub\\Test2\\IMG\\"
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

# Running Average Filter
# N is the filter order of mean average filter
N = 5 
window_array = np.ones(N)/N
SteerAngFiltMean = np.convolve(SteerAngRaw,window_array,mode = 'same')
SteerAngFiltMean = SteerAngFiltMean.tolist()

# Butterworth Filter
# Sampling Time measured from the simulator
'''
Filter parameters
passband frequency = 0.7 Hz = 0.7*14/2 = 0.1 
stopband frequency = 2 Hz = 3*14/2 = 0.3 
passband tolerance = 1 db
stopband attenuation = -60 db
'''
SampleTime = 0.07 # 142 images in 10 seconds
ButterOrder,CutOffFreq = signal.buttord(0.1,0.3,1,-60)
coNumButter, coDenButter = signal.butter(ButterOrder, CutOffFreq, 'low', \
                                         analog=False)
#w, h = scipy.signal.freqz(b, a)
SteerAngFiltButter = signal.filtfilt(coNumButter, coDenButter,SteerAngRaw)
SteerAngFiltButter = SteerAngFiltButter.tolist()

arrImages = []
arrSteerAng = []

for FileNameCenter,FileNameRight,FileNameLeft,SteerAng in zip(FileNamesCenter,FileNamesLeft,FileNamesRight,SteerAngRaw):
    
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
model.add(Dropout(0.3))
model.add(Dense(84))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('tanh'))

#sgd = optimizers.SGD(lr=0.001 )
adm = optimizers.adam(lr = 0.001)
model.compile(loss = "mse",optimizer = adm)
history = model.fit(X_train,y_train,validation_split = 0.2,shuffle = True\
                    ,epochs = 3,batch_size = 32)
					

model.save('model_test.h5')
    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


