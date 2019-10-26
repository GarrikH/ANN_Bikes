#!/usr/bin/env python
# coding: utf-8

# Step 1: Import modules

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend
from sklearn.model_selection import train_test_split
import numpy as np

# Step 2: Set our random seed

seed = 9
np.random.seed(seed)


# Step 3: Import our data set

from pandas import read_csv
filename = "BBCN.csv"
dataframe = read_csv(filename)
dataframe.head()

# Step 4: Split the Output Variables

array = dataframe.values

# Y will be the target variable

x = array[:, 0:11]
y = array[:, 11]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state = 9)


# Step 5: Build the Model
# First layer has 12 neurons and expects 11 input variables.
# The second hidden layer has 8 neurons,
# The output layer has 1 neuron to predict the class.

# kernel_initializer specifies the network weights, uniform makes them start as a small random number between 0 and .05

# relu = rectifier activation function

# sigmoid ensures output is 0 - 1

model = Sequential()
model.add(Dense(12, input_dim=11, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# Step 6: Compile the Model

# Metric is a function that is used to judge the performance of the model

# set the cost function to either categorical_crossentropy, binary_crossentropy is a cost functoin in logistic regression, categorical cross-entropy is generalizaiton for muli-class predictions via softmax

# binary is a log loss, binary - 0 or 1

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 7: Fit the Model

# epochs is iterations over the dataset;
# batch size is number of instances iterated over before weights are updated

model.fit(x, y, epochs=250, batch_size=15)

# Step 8: Score the model

scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

