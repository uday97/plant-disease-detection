from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import regularizers, optimizers
#from keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, Flatten, Dropout
from keras.layers import Dropout
#from keras.models import Model
import numpy as np
from keras.utils import np_utils
import keras
#import pandas as pd
import time
from scipy.misc import imread, imresize
import os
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir("./uploads"):
	if file.endswith(".jpg"):
		img = Image.open(dir_path+'/uploads/'+file)


	# In[3]:
def le_net(drop):
	model = Sequential()
	# first set of CONV => RELU => POOL
	model.add(Convolution2D(20, 5, 5, border_mode="same",
	input_shape=(60, 60, 3)))
	model.add(Dropout(drop))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))

	# second set of CONV => RELU => POOL
	model.add(Convolution2D(50, 5, 5, border_mode="same"))
	model.add(Dropout(drop))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(1000))
	model.add(Dropout(drop))
	model.add(Activation("relu"))

	# softmax classifier
	model.add(Dense(35))
	model.add(Dropout(drop))
	model.add(Activation("softmax"))

	return model

	# In[4]:

	# Setting up the hyperparameters
num_classes = 35
	#drop = 0.2
	# In[5]:

	# Initializing the model
model = le_net(0.2)

	# In[6]:

	# Load the weights of the trained model
model.load_weights('Lenet_all_60.h5')
	#train_images = np.load('train_all_images_lenet_60.npy')
	#train_labels = np.load('train_all_labels_lenet_60.npy')
	#labels = train_labels

	# In[7]:


	# Specify the class names

class_names = {0: 'AppleBlackRot', 1: 'AppleCedarAppleRust', 2: 'AppleHealthy', 3: 'AppleScab', 4: 'BlueberryHealthy',
	               5: 'CherryHealthy', 6: 'CherryPowderyMildew', 7: 'CornCommonRust', 8: 'CornHealthy', 9: 'CornNorthernLeafBlight',
	               10: 'GrapeBlackRot', 11: 'GrapeEsca', 12: 'GrapeHealthy', 13: 'GrapeLeafBlight', 14: 'OrangeHaunglongbing',
	               15: 'PeachBacterialSpot', 16: 'PeachHealthy', 17: 'PepperBellBacterialSpot', 18: 'PepperBellHealthy',
	               19: 'PotatoEarlyBlight', 20: 'PotatoLateBlight', 21: 'RaspberryHealthy', 22: 'SoybeanHealthy',
	               23: 'SquashPowderyMildew', 24: 'StrawberryHealthy', 25: 'StrawberryLeafScorch', 26: 'TomatoBacterialSpot',
	               27: 'TomatoEarlyBlight', 28: 'TomatoHealthy', 29: 'TomatoLateBlight', 30: 'TomatoLeafMold',
	               31: 'TomatoMosaicVirus', 32: 'TomatoSeptoriaLeafSpot', 33: 'TomatoTargetSpot', 34: 'TomatoYellowLeafCurlVirus'}

	
	# In[9]:


	#Testing random image from Apple Scab that is testing for one image.
	#mean = np.mean(train_images,axis=(0, 1, 2, 3)) gives value 117.530235
	#std = np.std(train_images,axis=(0, 1, 2, 3))   gives value 48.223305
mean = 117.530235
std = 48.223305
start_time = time.time()
	#img = imresize(imread(os.getcwd()+"/AppleScab/apple_scab_(5).jpg", mode='RGB'),(60,60)).astype(np.float32)
img = imresize(img,(60,60)).astype(np.float32)
img = (img-mean)/(std+1e-7)
img = np.expand_dims(img, axis=0)
out = model.predict(img)
finish_time = time.time()
time_diff2 = (finish_time - start_time) 

output = "Class-" + str(np.argmax(out)+1) + " : " + class_names[np.argmax(out)]
print(output)
	#print("time taken:"+str(time_diff2))
'''	
if __name__ == '__main__':
	main(img)
'''
	