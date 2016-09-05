local vision = require 'torchnet-vision.env'

local models = {}
vision.models = models

models.overfeat    = require 'torchnet-vision.models.overfeat'
models.vggm        = require 'torchnet-vision.models.vggm'
models.vgg16       = require 'torchnet-vision.models.vgg16'
models.inceptionv3 = require 'torchnet-vision.models.inceptionv3'
models.resnet      = require 'torchnet-vision.models.resnet'

return models
