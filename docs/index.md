## Authors

- Simone Bianco ([simone.bianco@disco.unimib.it](mailto:simone.bianco@disco.unimib.it)) - University of Milano-Bicocca<br>
- Claudio Cusano ([claudio.cusano@unipv.it](mailto:claudio.cusano@unipv.it)) - University of Pavia<br>

## Abstract

We present here a method for computational color constancy in which a deep convolutional neural network is trained to detect 
achromatic pixels in color images after they have been converted to grayscale.
The method does not require any information about the illuminant in the scene
and relies on the weak assumption, fulfilled by almost all images available on the Web, that training images have been approximately balanced.
Because of this requirement we define our method as *quasi-unsupervised*. 
After training, unbalanced images can be processed thanks to the preliminary conversion to grayscale of the input to the neural 
network.
The results of an extensive experimentation demonstrate that the proposed method is able to outperform the other unsupervised 
methods in the state of the art being, at the  same  time, flexible enough to be supervisedly fine-tuned to reach performance 
comparable with those of the best supervised methods.

## Overview

Given an unbalanced RAW image, the method estimates the color of the illuminant in the scene to make it possible to render the photo as if it was taken under a neutral illuminant.

<p align="center">
  <iframe frameborder="0" class="juxtapose" width="512" height="384" src="https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=6675ddfa-3a94-11e9-9dba-0edaf8f81e27"></iframe>
</p>

The illuminant is estimated as the weighted average of a set of pixels identified by a deep neural network trained on a large set 
of "almost balanced" images. The input is first converted to grayscale so that the neural network is not influenced by the real 
color of the illuminant ond so that the method can work equally well for both public images found on the web and for raw unbalanced images. This allows to leverage large training sets of images without having any information about the actual color of the illuminant.

![schema](https://raw.githubusercontent.com/claudio-unipv/quasi-unsupervised-cc/master/docs/schema.png)

Here are some other examples (input, selected pixels, balanced output).

![examples](https://raw.githubusercontent.com/claudio-unipv/quasi-unsupervised-cc/master/docs/examples-test-h.jpg)


## Installing and running the software

The method is implemented in the python programming language and uses the pytorch framework for deep learning.
It has been tested on a workstation equiped with a single NVIDIA Titan-Xp GPU and with the Ubuntu 18.04 operating system,
python version 3.6.7, CUDA 10.0, CUDNN 7.4.1.

To install the software the following steps are suggested (others may work as well):

from a terminal:
```
git clone https://github.com/claudio-unipv/quasi-unsupervised-cc.git
cd quasi-unsupervised-cc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src
```

When you finished using the software you can exit the virtual environment with the `deactivate` command.


### Training a new model

### Trained models

If you don't want to train the models by yourself, you can find some pretrained version at this 
[link](https://drive.google.com/drive/folders/1WYXCK-6rY4fxLnpXkJDd6h0-Dof_CLLG?usp=sharing).
Nine variants are provided differing in the training set (ILSVRC12, Places365, Flickr100k) and in
the information exploited (equalized grayscale, gradient directions, and their combination).

### Using an already trained model
