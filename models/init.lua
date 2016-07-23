local vision = require 'torchnet-vision.env'

local models = {}
vision.models = models

models.inceptionv3 = require 'torchnet-vision.models.inceptionv3'

return models
