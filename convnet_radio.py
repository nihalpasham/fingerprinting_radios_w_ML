
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import numpy as np

# Load data sets

# Training data 
Xtrain = np.load('train_samples.npy')
Ytrain = np.load('train_labels.npy')

# Test data
Xtest  = np.load('test_samples.npy')
Ytest  = np.load('test_labels.npy')

# Shuffle the data
Xtrain, Ytrain = shuffle(Xtrain, Ytrain)


# Define our network architecture:

# Input data is of the shape 2X128  
network = input_data(shape=[None, 2, 128, 1])

# Step 1: Convolution
network = conv_2d(network, 50, [1,3], activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Dropout - throw away some data randomly during training 
network = dropout(network, 0.25)

# Step 4: Convolution again
network = conv_2d(network, 50, [2,3], activation='relu')

# Step 5: Max pooling
network = max_pool_2d(network, 2)

# Step 6: Dropout - throw away some data randomly during training 
network = dropout(network, 0.25)

# Step 7: Fully-connected 256 node neural network
network = fully_connected(network, 256, activation='relu')

# Step 8: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 9: Fully-connected neural network with two outputs (0=sdr1, 1=sdr2) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='radio-identification.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(Xtrain, Ytrain, n_epoch=100, shuffle=True, validation_set=(Xtest, Ytest),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='radio-identification')

# Save model when training is complete to a file
model.save("radio-identification.tfl")
print("Network trained and saved as radio-identification.tfl!")