from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import random
import os
import shutil

class CreateDateset:
    def __init__(self, image_dir, label_dir, dataset_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.dataset_dir = dataset_dir
    
    def split_data(self, dir_name, ratio):
        """
        Split the smear dataset into two parts by ratio.
        """
        # Get all filenames.
        filenames = []
        files = os.listdir(dir_name)
        
        for file in files:
            if file.endswith('.BMP'):
                filename = file[0:-4]
                filenames.append(filename)
        part1, part2 = train_test_split(filenames, test_size=ratio, random_state=42)

        return part1, part2
    
    def get_paths(self, img_dir, label_dir, filenames):
        """
        Get paths of all images and labels of dataset.
        """
        image_paths = [image_dir + '/' + filename + '.BMP' for filename in filenames]
        label_paths = [label_dir + '/' + filename + '-d' + '.bmp' for filename in filenames]

        return image_paths, label_paths
    
    def copy_file(self, files, dir_name):
        for file in files:
            shutil.copy(file, dir_name)

    def create(self):
        # create test data
        train_val_data, test_data = self.split_data(self.image_dir, ratio=0.199)
        tr_val_img_paths, tr_val_label_paths = self.get_paths(self.image_dir, self.label_dir, train_val_data)
        test_img_paths, test_label_paths = self.get_paths(self.image_dir, self.label_dir, test_data)
        
        # copy files to dataset dir
        self.copy_file(tr_val_img_paths, self.dataset_dir + '/train_val/images')
        self.copy_file(tr_val_label_paths, self.dataset_dir + '/train_val/labels')
        self.copy_file(test_img_paths, self.dataset_dir + '/test/images')
        self.copy_file(test_label_paths, self.dataset_dir + '/test/labels')
        
        # create train and val data
        train_data, val_data = self.split_data(self.dataset_dir + '/train_val/images', ratio=0.249)
        tr_img_paths, tr_label_paths = self.get_paths(self.dataset_dir + '/train_val/images', self.dataset_dir + '/train_val/labels', train_data)
        val_img_paths, val_label_paths = self.get_paths(self.dataset_dir + '/train_val/images', self.dataset_dir + '/train_val/labels', val_data)
        
        # copy files to dataset_dir
        self.copy_file(tr_img_paths, self.dataset_dir + '/train/images')
        self.copy_file(tr_label_paths, self.dataset_dir + '/train/labels')
        self.copy_file(val_img_paths, self.dataset_dir + '/val/images')
        self.copy_file(val_label_paths, self.dataset_dir + '/val/labels')

# setup your own raw data folders
# use absolute path
image_dir = '/home/workspace/dataset/raw_data/images'
label_dir = '/home/workspace/dataset/raw_data/labels'
dataset_dir = '/home/workspace/dataset'
data = CreateDateset(image_dir, label_dir, dataset_dir)
data.create()
