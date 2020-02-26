# _*_ coding: utf-8 _*_

from core.network.ICPN import *
from core.data.get_batch_data import *

import keras.backend.tensorflow_backend as KTF

import math
import os 

os.environ["CUDA_VISIBLE_DECIVES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

data_dir = 'dataset/'

print('Loading test images...')
filenames = create_filenames_lists('test')
    
size_lists = []
image_paths = [data_dir + 'test/images/' + filename + '.BMP' for filename in filenames]
mask_paths = [data_dir + 'test/labels/' + filename + '-d' + '.bmp' for filename in filenames]

raw_images = [Image.open(image_path) for image_path in image_paths]
raw_masks = [Image.open(mask_path) for mask_path in mask_paths]
        
model = network(input_size=(256,256,3))
print('Loading model weights...')
model.load_weights("events/checkpoint/ICPN.hdf5")

print('There are {} pics to inference!'.format(len(raw_images)))
print('Predicting...')
for i in range(len(filenames)):
    img = Image.open(image_paths[i])
    mask = Image.open(mask_paths[i]).resize((256, 256))
    
    size = img.size
    img_ = img.resize((256, 256))
    image = np.asarray(img_)
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    
    logits = model.predict(image)
    pred = np.argmax(logits, axis=-1)
    pred = np.squeeze(pred)
    output = np.zeros(pred.shape + (3, ), dtype=np.uint8)
    for c, pal in enumerate(palette2):
        output[pred == c] = pal
        out = Image.fromarray(output).convert("RGB")
        # out = Image.fromarray(output).convert("RGB").resize(size, Image.NEAREST)
        out.save('predict/' + filenames[i] + '-pred.png')
    img_.save('predict/' + filenames[i] + '_raw.png')
    mask.save('predict/' + filenames[i] + '-d.png')

print('End!')
