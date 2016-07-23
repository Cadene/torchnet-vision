local vision = require 'torchnet-vision.env'

local model = {}
vision.model = model

model.inceptionv3 = require 'torchnet-vision.model.inceptionv3'

return model
