# torchnet-vision

*torchnet-vision* is a plugin for [torchnet](http://github.com/torchnet/torchnet) which provides a set
of abstractions aiming at encouraging code re-use as well as encouraging
modular programming.

At the moment, *torchnet-vision* provides several functionalities:
  - TransformImage: pre-processing image in various ways.
  - Datasets: download and load dataset easily.
  - Models: download and load models easily.  

For an overview of the *torchnet* framework, please also refer to [this paper](https://lvdmaaten.github.io/publications/papers/Torchnet_2016.pdf).


## Installation

Please install *torch* first, following instructions on
[torch.ch](http://torch.ch/docs/getting-started.html).  If *torch* is
already installed, make sure you have an up-to-date version of
[*argcheck*](https://github.com/torch/argcheck), otherwise you will get
weird errors at runtime.

Assuming *torch* is already installed, the *torchnet* and *torchnet-vision* cores are only a set of
lua files, so it is straightforward to install it with *luarocks*
```
luarocks install torchnet
git clone https://github.com/Cadene/torchnet-vision.git
cd torchnet-vision
luarocks make rocks/torchnet-vision-scm-1.rockspec
```


## Documentation

### Extraction features from lena with inceptionv3

```lua
require 'image'
tnt = require 'torchnet'
vision = require 'torchnet-vision'

augmentation = tnt.transform.compose{
   vision.image.transformimage.randomScale{minSize=299,maxSize=350},
   vision.image.transformimage.randomCrop(299),
   vision.image.transformimage.colorNormalize{
      mean = vision.models.inceptionv3.mean,
      std  = vision.models.inceptionv3.std
   },
   function(img) return img:float() end
}
img = augmentation(image.lena())

net = vision.models.inceptionv3.loadExtracting{ -- download included
   filename = 'tmp/inceptionv3.t7',
   layerid  = 30
}
net:evaluate()
print(net:forward(img:view(1,3,299,299)):size()) -- 2048
```

### Fine tuning on MIT67 in 250 lines of code

```
$ CUDA_VISIBLE_DEVICES=0 th demo/mainmit67.lua -usegpu true
$ ls demo/logs/mit67/*/
$ cat demo/logs/mit67/*/trainlog.txt
$ cat demo/logs/mit67/*/testlog.txt
```  

## Other projects using torchnet

- [Wide residual network](https://github.com/szagoruyko/wide-residual-networks)
- [MultiGPU ImageNet](https://github.com/karandwivedi42/imagenet-multiGPU.torchnet)
- [Learning to Compare Image Patches via Convolutional Neural Networks (CVPR 2015)](https://github.com/szagoruyko/cvpr15deepcompare/blob/master/training/train.lua)
- [Facebook "A MultiPath Network for Object Detection"](https://github.com/facebookresearch/multipathnet)

## Roadmap

- defining names for package and classes (vision?)
- add docs to TransformImage methods
- add test
- add a method to tnt.DataIterator to process the mean and std
- add a better system to preprocess images than tnt.transform (especially to add or remove TransformImage.colorNormalize)
- add data loaders (largscale or not)
- add video directory
