# _*_ coding: utf-8 _*_

from  keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
from PIL import Image

# setup you own train, val and test dataset folders
# use absolute path
data_dir = '/home/workspace/ICPN/dataset/'

## Label palette config
# 2 classes palette
background = [0, 0, 0] # 4-1
nuclei = [255 , 255, 255] # 2-0

palette2 = [nuclei,
        background]


def create_filenames_lists(split):
    """
    Get filenames of images and labels
    """
    
    if split == 'train':
        image_files = os.listdir(data_dir + split + '/images')
        filenames = [image_file[0:-4] for image_file in image_files if image_file.endswith('.BMP')]
        
    if split == 'val':
        image_files = os.listdir(data_dir + split + '/images')
        filenames = [image_file[0:-4] for image_file in image_files if image_file.endswith('.BMP')]
    
    if split == 'test':
        image_files = os.listdir(data_dir + split + '/images')
        filenames = [image_file[0:-4] for image_file in image_files if image_file.endswith('.BMP')]
    
    return filenames


def adjust_2classes(image, mask, nClass, palette):
    """
    image normalizing and one hot encoding for label.
    """
    image = image / 255.
    
    mask = np.squeeze(mask, axis=3)
    mask = mask.astype(dtype=np.uint8)
    if palette is not None:
        one_hot_maps = np.zeros((mask.shape + (nClass, )), dtype=np.float32)
        for c in range(nClass):
            if c == 0:
                mask_arr = mask == 2
                one_hot_maps[:, :, :, c][mask_arr] = 1
            elif c == 1:
                mask_arr = mask == 4
                one_hot_maps[:, :, :, c][mask_arr] = 1
                mask_arr = mask == 3
                one_hot_maps[:, :, :, c][mask_arr] = 1
                mask_arr = mask == 1
                one_hot_maps[:, :, :, c][mask_arr] = 1
                mask_arr = mask == 0
                one_hot_maps[:, :, :, c][mask_arr] = 1
    
    return image, one_hot_maps

    
def trainGenerator(batch_size, nClass, data_gen_args, palette=None, target_size=[256, 256], save_to_dir=None):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    """
    # Read images and labels
    filenames = create_filenames_lists('train')
    
    image_paths = [data_dir + 'train/images/' + filename + '.BMP' for filename in filenames]
    mask_paths = [data_dir + 'train/labels/' + filename + '-d' + '.bmp' for filename in filenames]
    num_files = len(image_paths)
    
    raw_images = [Image.open(image_path) for image_path in image_paths]
    raw_masks = [Image.open(mask_path) for mask_path in mask_paths]
    
    # resize to 255 * 255
    resize_images = [image.resize((target_size[0], target_size[1])) for image in raw_images]
    resize_masks = [mask.resize((target_size[0], target_size[1])) for mask in raw_masks]
    
    images = np.array([list(np.asarray(image)) for image in resize_images])
    masks = np.array([list(np.asarray(mask)) for mask in resize_masks])
    masks = np.array([list(np.expand_dims(mask, axis=3)) for mask in masks])
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    
    image_generator = image_datagen.flow(images, seed=seed, batch_size=batch_size, shuffle=True, save_to_dir=save_to_dir)
    
    mask_generator = mask_datagen.flow(masks, seed=seed, batch_size=batch_size, shuffle=True, save_to_dir=save_to_dir)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = adjust_2classes(img, mask, nClass, palette)
        
        yield (img, mask)
        

def valGenerator(batch_size, nClass, palette=None, target_size=[256, 256], save_to_dir=None):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    """
    # Read images and labels
    filenames = create_filenames_lists('val')
    
    image_paths = [data_dir + 'val/images/' + filename + '.BMP' for filename in filenames]
    mask_paths = [data_dir + 'val/labels/' + filename + '-d' + '.bmp' for filename in filenames]
    num_files = len(image_paths)
    
    raw_images = [Image.open(image_path) for image_path in image_paths]
    raw_masks = [Image.open(mask_path) for mask_path in mask_paths]

    resize_images = [image.resize((target_size[0], target_size[1])) for image in raw_images]
    resize_masks = [mask.resize((target_size[0], target_size[1])) for mask in raw_masks]
    
    images = np.array([list(np.asarray(image)) for image in resize_images])
    masks = np.array([list(np.asarray(mask)) for mask in resize_masks])
    masks = np.array([list(np.expand_dims(mask, axis=3)) for mask in masks])
    
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    seed = 1
    image_datagen.fit(images, seed=seed)
    mask_datagen.fit(masks, seed=seed)
    
    image_generator = image_datagen.flow(images, seed=seed, batch_size=batch_size)
    
    mask_generator = mask_datagen.flow(masks, seed=seed, batch_size=batch_size)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = adjust_2classes(img, mask, nClass, palette)
        
        yield (img, mask)
        
        
def testGenerator(batch_size, nClass, palette=None, target_size=[256, 256]):
    """
    Get batch of test data for testing.
    """
    # Read images and labels
    filenames = create_filenames_lists('test')
    
    image_paths = [data_dir + 'test/images/' + filename + '.BMP' for filename in filenames]
    mask_paths = [data_dir + 'test/labels/' + filename + '-d' + '.bmp' for filename in filenames]
    num_files = len(image_paths)
    
    raw_images = [Image.open(image_path) for image_path in image_paths]
    raw_masks = [Image.open(mask_path) for mask_path in mask_paths]
    
    # Resize images and masks
    resize_images = [image.resize((target_size[0], target_size[1])) for image in raw_images]
    resize_masks = [mask.resize((target_size[0], target_size[1])) for mask in raw_masks]
    
    images = np.array([list(np.asarray(image)) for image in resize_images])
    masks = np.array([list(np.asarray(mask)) for mask in resize_masks])
    masks = np.array([list(np.expand_dims(mask, axis=3)) for mask in masks])
    
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    seed = 1
    image_datagen.fit(images, seed=seed)
    mask_datagen.fit(masks, seed=seed)
    
    image_generator = image_datagen.flow(images, seed=seed, batch_size=batch_size)
    mask_generator = mask_datagen.flow(masks, seed=seed, batch_size=batch_size)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = adjust_2classes(img, mask, nClass, palette)
        
        yield (img, mask)
        
        
def testImage(raw_images, target_size=[256, 256]):
    """
    Get test images for predict.
    """
    resize_images = [image.resize((target_size[0], target_size[1])) for image in raw_images]
    images = np.array([list(np.asarray(image)) for image in resize_images])
    
    image_datagen = ImageDataGenerator()
    seed = 1
    image_generator = image_datagen.flow(images, seed=seed, batch_size=1)
    
    for img in image_generator:
        img = img / 255.
        yield img
