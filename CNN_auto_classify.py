"""Autoanalyization of different CNN-Models with TensorBoard (and tensorflow-gpu)"""

# Import of all relevant packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from tensorflow.keras.callbacks import TensorBoard 
import time 
            
#%% Define gpu workload, if training different models parallel
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#%% Import data (already preprocessed)
X = pickle.load(open("X.pickle","rb")) # Pictures of cats and dogs
y = pickle.load(open("y.pickle","rb")) # Corresponding lasses of cats and dogs

X = np.array(X) # conversion to array
y = np.array(y)

X = X/255 # Data normalization

#%% Create model
# Create model parameters
conv_layers = [1, 2, 3] 
layer_sizes = [32, 64, 128]
dense_layers = [0, 1, 2]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            # Individual name per model
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            
            # Tensorboard init
            tensorboard = TensorBoard(log_dir="C:\\temp\\tensorflow_logs\\{}".format(NAME))
            
            # Modell init
            model = Sequential()
            
            # 1.Layer
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:])) # 1.Layer needs input shape
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dropout(0.2)) # prevents the model from overfitting
            
            # Output Layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            # Train model
            model.fit(X, y, batch_size=32, epochs=1, validation_split=0.3, callbacks=[tensorboard])
            
 """How to call tensorboard
1. Check if logs are created in "C:\temp\tensorflow_logs\..."
2. Start cmd and navigate to "C:\temp"
3. Activate environment where tensorflow is installed (e.g. (deepml) C:\temp>tensorboard --logdir=tensorflow_logs\).
4. Call http://localhost:6006/ in browser and start analyzing the different models in realtime."""
