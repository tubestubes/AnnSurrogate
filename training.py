"""
This script trains an ANN to model the function of 7 RVs given in fn.py
* Requires data:
- dtest.pickle
- dtrain.pickle
"""


"""__Preamble______________________________________________________________________________________________________"""
# Tensforflow settings
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      # Run this line to force CPU on CUDA enabled GPUs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'        # =1 for no TS into, =0 for default TS info

import pickle
import time
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


"""__Pre-Processing________________________________________________________________________________________________"""
print("Importing data...")
# Pickle in training and test data
with open('dtrain.pickle', 'rb') as file:
    d_train = pickle.load(file)
with open('dtest.pickle' , 'rb') as file:
    d_test = pickle.load(file)
#print(d_train.head()) # Use to test import

# Pop out labels
l_train = d_train.pop('label')
l_test  = d_test.pop('label')


"""__Modelling____________________________________________________________________________________________________"""
print("Building model...")
EPOCHS = 1000
# Set up model architecture
model = keras.Sequential([
            layers.Dense(48, activation='relu', input_shape=[7]),
            layers.Dense(48, activation='relu'),
            layers.Dense(48, activation='relu'),
            layers.Dense(1)
            ])
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
# This will stop the model early if it doesnt improve much in the last 10 epochs
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train the model
print("Training Model...")
# verbose=2 to see progress of model during training, or =0 for silence
# Validation split creates test data for work-in-progress performance stats, not used for training
t_start = time.time()
model.fit(d_train, l_train, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop])
t_fin = time.time()
print("Done. Computation time (seconds):", t_fin - t_start)

# Export model for analysis
#model.save('ann_model')

# Evaluate performance on test data
loss, mae, mse = model.evaluate(d_test, l_test, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} ".format(mae))