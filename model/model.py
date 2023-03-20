import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os 


#The following two lines limits tensorflow from using all the gpus on our machine. 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

