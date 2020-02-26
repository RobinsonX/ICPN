from core.network.ICPN import *

from core.data.get_batch_data import *

import tensorflow as tf

import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.utils import plot_model

import math
import os 

# gpu setup
os.environ["CUDA_VISIBLE_DECIVES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

# dice coeff
def dice_coeff(y_true, y_pred):
    
    smooth = 1
    
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# dice loss
def dice_loss(y_true, y_pred):
    
    return 1-dice_coeff(y_true, y_pred)

# data aug
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

# number of val images
valNum = len(create_filenames_lists('val'))

# train data and validation data
trainDate = trainGenerator(4, 2, data_gen_args, palette=palette2, target_size=[256, 256], save_to_dir=None)
valDate = valGenerator(4, 2, palette=palette2, target_size=[256, 256], save_to_dir=None)

# Get Model
model = network(input_size=(256,256,3))

# Optimizer
model.compile(optimizer=Adam(lr=1e-3, decay=0.0005), loss=[dice_loss], metrics=[dice_coeff])

# plot model
plot_model(model, to_file='events/graphviz/ICPN.png')
model.summary()

# Callbacks
model_checkpoint = ModelCheckpoint('events/checkpoint/ICPN.hdf5', monitor='loss',verbose=1, save_best_only=True, period=1)
early_stopping = EarlyStopping(monitor='loss',patience=2)
tensorboard = TensorBoard(log_dir='events/tensorboard/ICPN/', write_images=False)
csvlog = CSVLogger("events/csvlog/ICPN.log")

# Load Model
if os.path.exists('events/checkpoint/ICPN.hdf5'):
    model.load_weights('events/checkpoint/ICPN.hdf5')
    print("checkpoint_loaded")
    
print("network training...")

# Visualizing training
history = model.fit_generator(trainDate,steps_per_epoch=2000,epochs=30,callbacks=[model_checkpoint, early_stopping, tensorboard], validation_data=valDate, validation_steps=math.ceil(valNum / 4))
