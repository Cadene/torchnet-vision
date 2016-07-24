local vision = require 'torchnet-vision.env'

local models = {}
vision.models = models

models.vgg16       = require 'torchnet-vision.models.vgg16'
models.inceptionv3 = require 'torchnet-vision.models.inceptionv3'

return models
