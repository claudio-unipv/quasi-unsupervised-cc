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

## Paper

Please include the following reference in your paper if you mention the method: 

```
@inproceedings{quasiunsupervisedcc2019,
  title= = {Quasi-unsupervised color constancy},
  author = {Bianco, Simone and Cusano, Claudio},
  booktitle = {CVPR-2019},
  year = {2019}
}
```

The [poster presentation](https://github.com/claudio-unipv/quasi-unsupervised-cc/raw/master/docs/poster.pdf) and some [additional material](https://github.com/claudio-unipv/quasi-unsupervised-cc/raw/master/docs/Quasi-unsupervised-additional-material.pdf) are also available.


## Overview

Given an unbalanced RAW image, the method estimates the color of the illuminant in the scene to make it possible to render the photo as if it was taken under a neutral illuminant.

<p align="center">
<iframe frameborder="0" class="juxtapose" width="512" height="342" src="https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=8f3344e4-3acb-11e9-9dba-0edaf8f81e27"></iframe>
</p>

The illuminant is estimated as the weighted average of a set of pixels identified by a deep neural network trained on a large set 
of "almost balanced" images. The input is first converted to grayscale so that the neural network is not influenced by the real 
color of the illuminant and so that the method can work equally well for both public images found on the web and raw unbalanced images. This allows to leverage large training sets of images without having any information about the actual color of the illuminant.

![schema](https://raw.githubusercontent.com/claudio-unipv/quasi-unsupervised-cc/master/docs/schema.png)

The neural network is a U-Net with 8 convolution/deconvolution pairs and skip connections.

![schema](https://raw.githubusercontent.com/claudio-unipv/quasi-unsupervised-cc/master/docs/architecture-h.png)

Here are some other examples (input, selected pixels, balanced output).

![examples](https://raw.githubusercontent.com/claudio-unipv/quasi-unsupervised-cc/master/docs/examples-test-h.jpg)




## Interactive Demo

A running version of the method is [available online](http://democusano.unipv.it:80/demo).


## Installing and running the software

The method is implemented in the python programming language and uses the pytorch framework for deep learning.
It has been tested on a workstation equiped with a single NVIDIA Titan-Xp GPU and with the Ubuntu 18.04 operating system,
python version 3.6.7, CUDA 10.0, CUDNN 7.4.1.

To install the software the following steps are suggested (others may work as well).

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


### Processing the images

To apply the method to images you need a trained model.  If you don't
want to train it by yourself, you can find
[here](https://drive.google.com/drive/folders/1WYXCK-6rY4fxLnpXkJDd6h0-Dof_CLLG?usp=sharing)
some pretrained version.  Nine variants are provided differing in the
training set (ILSVRC12, Places365, Flickr100k) and in the information
exploited (equalized grayscale, gradient directions, and their
combination).

The paths to the images to process must be placed in a text
file (one path per line).  Optionally, the path can be followed by the
RGB components of the actual color of the illuminant.  This allows to
evaluate the accuracy of the model.  See, for instance, the lists in
the `data` and `examples` directories.

The `evalmodel.py` script, in the `src` directory will apply the model.  For instance

```
python3 evalmodel.py --apply-gamma --output-dir out ilsvrc12-eg.pt ../examples/examples.txt
```

This will apply the model `ilsvrc12-eg.pt` (which is the recommended
one) to the images listed in `../examples/examples.txt` and will save
the results in the directory `out`.  The option `--apply-gamma`
applies the sRGB gamma to the balanced output images.  See the output
of `python3 evalmodel.py -h` for more options.


### Training a new model

Similarly to the testing procedure, traininig a new model requires 
that the paths to the images are listed in a text file to form a training set.

The command to give is then:
``` 
python3 trainmodel.py --training-list train.txt model_dir
```
where `train.txt` is the file listing the training images and `model_dir` is a directory
where output files are placed.  The training script supports many options (type
`python3 trainmodel.py -h` to list them).  Default values are those used in the paper
for the "quasi-unsupervised" setup.

For supervised training it is enough to use an annotated training set (i.e. one where each
path is followed by the ground thruth illuminant).  However, if the training set is small
it is better to start from a model pretrained in the quasi-unsupervised setup. 
