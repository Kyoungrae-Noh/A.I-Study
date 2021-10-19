# ESPCN in Tensorflow

Tensorflow implementation of [Efficient Sub-Pixel Convolutional Neural Networks for Single Image Super-Resolution](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)[1].

It was trained on the [Div2K datset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) - Train Data (HR images).

## Requirements
-Tensorflow
- numpy
- cv2

## ESPCN
This is th ESPCN model with two convolution layers for feature maps extraction, and a sub-pixel convolution layer that aggregates the feature map from LR space and builds the SR image in a single step.

![Alt text](images/ESPCN.png?raw=true "ESPCN architecture")

## Example
