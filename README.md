# INTERACTING CONVOLUTION WITH PYRAMID STRUCTURE NETWORK FOR AUTOMATED SEGMENTATION OF CERVICAL NUCLEI IN PAP SMEAR IMAGES
By Xiaoqing Yang, Junmin Wu, Yan Yin.

## Introduction
We propose an Interacting Convolution with Pyramid Structure Network (ICPN), which consists of a sufficient aggregation path that focus on more nucleus contexts and a selecting path that enable nucleus localization. The two paths are built on Interacting Convolutional Modules (ICM) and Internal Pyramid Resolution Complementing Modules (IPRCM) respectively. We evaluate our network on public [Herlev dataset(Part II-smear2005.zip)](http://mde-lab.aegean.gr/index.php/downloads).

## Architecture
![](img/ICPN.PNG)

Overview of the proposed ICPN for cervical nuclei nucleus segmentation. ICPN consits of two paths: the aggregating path and selecting path. The former has N encoders and the latter has N decoders. There are no additional skip connections between two paths.

## Interacting Convolutional Module
![](img/ICM.PNG)

Interacting Convolutional Module has two parallel convolution paths. The upper path uses a smaller convolution kernel to capture detailed structural information inside the nucleus, while the lower path uses a relatively large convolution kernel to catch the global context of the overall structure of the nucleus. There is an information exchange in the process. Similar to the GCN, we use the similar approach to reduce parameters (batch normalization, separate kernels). 

![](img/IPRCM.PNG)

Internal Pyramid Resolution Complementing Module converts the features of the current resolution representations and restores these multiresolution features to the current resolution size. This process captures the potential multiresulution representations of the current resolution features, thereby compensating for the loss of information caused by the pooling layer. The 1*1 convolutional layer is used for dimensionality reduction.

## Herlev testing result
![](img/result.PNG)

## Usage
### Requirements
You can setup you envirnoment by requirements.txt.
To install tensorflow == 1.12.0, please refer to [](https://www.tensorflow.org/install)
To install keras == 2.2.0, please refer to [](https://keras.io/#installation)

### Dataset splitting
 1. Download smear2005 dataset.
 2. copy all cervical images into "dataset/raw_data/images".
 3. copy all cervical labels into "dataset/raw_date/labels".
 4. setup your absolute paths in "dataset/create_dataset.py".
 5. do following commands:
    ```
     cd dataset/
     python create_dataset.py
    ``` 
