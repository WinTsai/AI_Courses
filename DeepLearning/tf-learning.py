# this is exercise file of tensorflow file
# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init
model = keras.Sequential()
model.add(layers.Dense(1,kernel_initializer=init.RandomNormal(stddev=0.01)))


