import tensorflow
import os 

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tensorflow.config.experimental.set_memory_growth(gpu, True)