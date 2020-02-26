# _*_ coding: utf-8 _*_

from core.network.ICPN import *
from core.data.get_batch_data import *

# gpu setup
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DECIVES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

# recall
def recall_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    
    return recall

# precision
def precision_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    
    return precision

# f1 score
def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

# test data
testGene = testGenerator(32, 2, palette=palette2, target_size=[256, 256])
    
# Get Model
model = network(input_size=(256,256,3))

model.compile(optimizer=Adam(lr=1e-3, decay=0.0005), loss=[dice_loss], metrics=[dice_coeff, precision_m, recall_m, f1_m])

# Load Model
print('Loading model weights...')
model.load_weights("events/checkpoint/ICPN.hdf5")

print('Testing...')
scores = model.evaluate_generator(testGene, steps=6, verbose=0)
print('Test loss:', scores[0])
print('Test dice:', scores[1])
print('Test pre:', scores[2])
print('Test recall:', scores[3])
print('Test f1:', scores[4])
print('End!')
