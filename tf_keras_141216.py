# This is a expanded code follow through of the following tutorial:
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python

# I use Tensorflow for backend. The tutorial uses Theano.

# I try to explain some of the stuff line by line, for my own benefit.


import numpy as np
import tensorflow as tf
np.random.seed(123) # For reproducibility


# Set backend to Tensorflow
from keras import backend as K
K.set_image_dim_ordering('tf')


# This model is a linear stack of neural network layers 
# It's perfect for the type of feed-forward CNN we're building

from keras.models import Sequential 

# Import "core" layers from Keras. 
# These are the layers that are used in almost any neural network

from keras.layers import Dense, Dropout, Activation, Flatten

# Import the convolutional neural network (CNN) layers
# They will help us train on image data
# There's a ton of other layers in Keras
# You can even write your own!

# Nah, we'll use what they already have

from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU

# Load some utilities to transform the data

from keras.utils import np_utils

# Supposed to use the csv provided by Kaggle
# But this is way easier to load, and the data should be the same
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# We must explicitly declare a dimension for the depth of the input image. 
# For example, a full-color image with all 3 RGB channels will have a depth of 3.

# The MNIST images only have a depth of 1, but we must explicitly declare that.

# In other words, we want to transform our dataset from having shape (n, width, height) 
# to (n, width, height, depth). The tutorial has it the other way around "because Theano"

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
 
# The final preprocessing step for the input data is to convert our data type 
# We need float32 type and normalize our data values to the range [0, 1]


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255  

# The labels are imported as a 1-dim array
# Must be 10-dim, for each class

Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)

"""
$$$ Done with pre-processing, now we start modeling $$$

"""
 
# Declare the model
model = Sequential()

# The tutorial had the arguments for input_shape the other way around, because 
# it uses Theano for backend. I am using TF, and the depth dimension argument comes last.

# Keras Docs:

# Input shape: 4D tensor with shape: (samples, channels, rows, cols) if dim_ordering='th'
# or 4D tensor with shape: (samples, rows, cols, channels) if dim_ordering='tf'.

# We also define which activation function to use. The tutorial uses 'relu'(rectified linear unit)
# This means we're using Rectifier as the activation function. Rectifier is defined as
# f(x)=max(0,x), where x is the input to a neuron. 

# You may know a more common activation function: sigmoid. There is a good explanation
# why ReLU is better suited for NNs on Stack Overflow. I need my math wetware upgraded
# to understand the advantages clearly, but feel free to dive in.

# The first 3 parameters represent the number of convolution filters to use, 
# the number of rows in each convolution kernel, 
# and the number of columns in each convolution kernel, respectively.

# I understand it to be the makeup of the network. As with any parameters,
# I assume they can be tweaked to squeeze out better performance later.

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))

# Now we stack those layers.
# Note the Dropout layer. This is a method for regularizing the model to prevent overfitting. 

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # Goal: reduce the number of model params 
model.add(Dropout(0.25)) # Means we discard 25% of the data cause we're balling over here

# The weights from the Convolution layers must be flattened (made 1-dimensional) 
# before passing them to the fully connected Dense layer.

model.add(Flatten())
model.add(Dense(128, activation='relu')) # First param = output size. 32 filters * 2*2 
model.add(Dropout(0.5)) # Dropping out again, now even more
model.add(Dense(10, activation='softmax')) # 10 outputs = 10 classes. This is the final layer.

"""

Compile

"""

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
          
"""

Fit

"""
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=3, verbose=1)

