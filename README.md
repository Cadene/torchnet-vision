# torchnet-vision

*torchnet-vision* is a plugin for [torchnet](http://github.com/torchnet/torchnet) which provides a set
of abstractions aiming at encouraging code re-use as well as encouraging
modular programming.

At the moment, *torchnet-vision* provides one class:
  - TransformImage: pre-processing image in various ways.

For an overview of the *torchnet* framework, please also refer to [this paper](https://lvdmaaten.github.io/publications/papers/Torchnet_2016.pdf).


## Installation

Please install *torch* first, following instructions on
[torch.ch](http://torch.ch/docs/getting-started.html).  If *torch* is
already installed, make sure you have an up-to-date version of
[*argcheck*](https://github.com/torch/argcheck), otherwise you will get
weird errors at runtime.

Assuming *torch* is already installed, the *torchnet* core is only a set of
lua files, so it is straightforward to install it with *luarocks*
```
luarocks install torchnet
luarocks install image
git clone https://github.com/Cadene/torchnet-vision.git
cd torchnet-vision
luarocks make rocks/torchnet-vision-scm-1.rockspec
```


## Documentation

```lua
require 'image'
local vision = require 'torchnet-vision'
local transf = vision.TransformImage()
print(transf:randomScale{minSize=200,maxSize=300}(image.lena()):size())

```


## Roadmap

- defining names for package and classes (vision?)
- keep TransformImage non-static ?
- add docs to TransformImage methods
- add test
- add a method to tnt.DataIterator to process the mean and std
- add a better system to preprocess images than tnt.transform (especially to add or remove TransformImage.colorNormalize)
- add data loaders (largscale or not)
- add video directory
