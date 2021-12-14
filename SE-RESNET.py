import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
#tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)

# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()
import numpy as np
import statistics
from numpy import *
import matplotlib.pyplot as plt
import csv
from PIL import Image
from sklearn import preprocessing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.sequence import *
from collections import defaultdict
from collections import Counter

import keras
from keras import backend as K
from keras.models import Model

from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
#from keras.layers import Activation, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, AveragePooling2D
from keras.layers.pooling import MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.activations import relu
import cv2
from tensorflow.python.keras.utils import np_utils
from keras import backend as K
#K.common.image_dim_ordering()
#K.set_image_dim_ordering('th')
from keras.utils.vis_utils import plot_model
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose,Input

from keras.layers import Dense
from keras.regularizers import l2, l1
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.callbacks import ModelCheckpoint
#from imageGenerator1 import ImageDataGenerator
#from imagegen_csv import ImageDataGenerator
from Generator import ImageDataGenerator

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
batch_size=5
datagen = ImageDataGenerator()
#train_data = datagen.flow_from_directory('D:/HR_estimation/ROI1', 'D:/HR_estimation/heart_rate', target_size=(120, 120), class_mode='label', batch_size=1, frames_per_step=75, shuffle=False)
#test_data=datagen.flow_from_directory('D:/HR_estimation/ROI2', 'D:/HR_estimation/heart_rate2', target_size=(120, 120), class_mode='label', batch_size=1, frames_per_step=75)
train_data = datagen.flow_from_directory('/home/ouzar1/Desktop/Dataset1/ROI', '/home/ouzar1/Desktop/Dataset1/HR', target_size=(260, 348), class_mode='label', batch_size=batch_size, frames_per_step=75, shuffle=False)
test_data = datagen.flow_from_directory('/home/ouzar1/Desktop/Dataset1/MMSE1', '/home/ouzar1/Desktop/Dataset1/MMSE-HR1', target_size=(260, 348), class_mode='label', batch_size=1, frames_per_step=75, shuffle=False)
print("finished1")
from keras import optimizers, losses
from keras.layers import *
from keras.models import Model
from keras.backend import int_shape
from keras.utils import to_categorical, plot_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def se_block(block_input, num_filters, ratio=8):  # Squeeze and excitation block

    '''
        Args:
            block_input: input tensor to the squeeze and excitation block
            num_filters: no. of filters/channels in block_input
            ratio: a hyperparameter that denotes the ratio by which no. of channels will be reduced

        Returns:
            scale: scaled tensor after getting multiplied by new channel weights
    '''

    pool1 = GlobalAveragePooling3D()(block_input)
    flat = Reshape((1, 1, 1, num_filters))(pool1)
    dense1 = Dense(num_filters // ratio, activation='relu')(flat)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = multiply([block_input, dense2])

    return scale


def resnet_block(block_input, num_filters):  # Single ResNet block

    '''
        Args:
            block_input: input tensor to the ResNet block
            num_filters: no. of filters/channels in block_input

        Returns:
            relu2: activated tensor after addition with original input
    '''

    if int_shape(block_input)[3] != num_filters:
        block_input = Conv3D(num_filters, kernel_size=(1, 1, 1))(block_input)

    conv1 = Conv3D(num_filters, kernel_size=(3, 3, 3), padding='same')(block_input)
    norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(norm1)
    conv2 = Conv3D(num_filters, kernel_size=(3, 3, 3), padding='same')(relu1)
    norm2 = BatchNormalization()(conv2)

    se = se_block(norm2, num_filters=num_filters)

    sum = Add()([block_input, se])
    relu2 = Activation('relu')(sum)

    return relu2


def se_resnet14():
    '''
        Squeeze and excitation blocks applied on an 14-layer adapted version of ResNet18.
        Adapted for MNIST dataset.
        Input size is 28x28x1 representing images in MNIST.
        Output size is 10 representing classes to which images belong.
    '''

    input = Input(shape=(75, 260, 348, 3))
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    pool1 = AveragePooling3D((1, 2, 2), strides=2)(conv1)

    block1 = resnet_block(pool1, 32)
    block2 = resnet_block(block1, 32)

    pool2 = AveragePooling3D((1, 5, 6), strides=2)(block2)

    block3 = resnet_block(pool2, 32)
    block4 = resnet_block(block3, 32)

    pool3 = AveragePooling3D((3, 1, 1), strides=2)(block4)

    block5 = resnet_block(pool3, 64)
    block6 = resnet_block(block5, 64)

    pool4 = AveragePooling3D((5, 5, 5), strides=2)(block6)

    block7 = resnet_block(pool4, 64)
    block8 = resnet_block(block7, 64)

    pool5 = AveragePooling3D((2, 2, 2), strides=2)(block8)
    flat = Flatten()(pool4)
    x = Dense(1024, activation='relu')(flat)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=input, outputs=output)
    return model




model = se_resnet14()

epochs = 25
drop_rate = 0.1
lr = 0.01
#model = densenet_3d(1, input_shape, dropout_rate=drop_rate)
#model = resnet(input_shape)
#model = CNNModel()
opt = Adam(lr=0.0001, decay=0.05)
opt1 = tf.keras.optimizers.Adamax(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")
rmse = tf.keras.metrics.RootMeanSquaredError()
#run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
print("finished2")


model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', rmse])

sgd = SGD(lr=lr, momentum=0.9, nesterov=True)

model.summary()

print('Start training..')

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
import pandas as pd

# every epoch check validation accuracy scores and save the highest
checkpoint_2 = ModelCheckpoint('weights-{epoch:02d}.h5',
                               monitor='val_root_mean_squared_error',
                               verbose=1, save_best_only=False)
# every 10 epochs save weights
checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_root_mean_squared_error:.4f}.h5',
                             monitor='val_mean_absolute_error',
                             verbose=1, save_best_only=False)
history_checkpoint = CSVLogger("SE_RESNET.csv", append=True)

# use tensorboard can watch the change in time
tensorboard_ = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=5, verbose=1, mode='auto')
"""
if (CONTINUE_TRAINING == True):
    history = pd.read_csv('history2.csv')
    INITIAL_EPOCH = history.shape[0]
    model.load_weights('weights_%02d.h5' % INITIAL_EPOCH)
    checkpoint_2.best = np.min(history['val_root_mean_squared_error'])
else:
    INITIAL_EPOCH = 0
"""
history = model.fit_generator(train_data, epochs=100,
                                  steps_per_epoch= len(train_data.filenames) // 375,
                                  validation_data=test_data, validation_steps=len(test_data.filenames) // 75,
                                callbacks=[checkpoint, history_checkpoint])

values = history.history
validation_loss = values['val_loss']
validation_mae = values['val_mae']
training_mae = values['mae']
validation_rmse = values['val_root_mean_squared_error']
training_rmse = values['root_mean_squared_error']
training_loss = values['loss']

epochs = range(100)

plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.title('Epochs vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, training_mae, label='Training MAE')
plt.plot(epochs, validation_mae, label='Validation MAE')
plt.title('Epochs vs MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

plt.plot(epochs, training_rmse, label='Training RMSE')
plt.plot(epochs, validation_rmse, label='Validation RMSE')
plt.title('Epochs vs RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()
plt.show()

# scores = model.evaluate(train_data)
scores = model.predict_generator(test_data, len(test_data.filenames) // 75)
print("%s: %.2f" % (model.metrics_names[1], scores[1]), "%s: %.2f" % (model.metrics_names[2], scores[2]))