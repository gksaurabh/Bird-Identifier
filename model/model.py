import tensorflow as tf
import numpy as np
import matplotlib
import os 
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensroflow.keras.models import Conv2D, Maxpooling2D, Dense, Flatten, Dropout


model = Sequential()

#Adding a convulutional layer and a maxpooling layer to the model


#Layer 1 
	#Conv2D parameters, 16 filters, 3x3 pixels with stride length 1 with the Relu activation with specified input size
	#Then use the maxpooling layer to get the maxvalue of a 2x2 region for each region 
model.add(Conv2D(16, (3,3),1, activation='relu', input_shape=(224,224,3)))
model.add(Maxpooling2D())

#Layer 2
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(Maxpooling2D())

#Layer 3
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(Maxpooling2D())

#Layer 4
model.add(Flatten())

#Layer 5
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))