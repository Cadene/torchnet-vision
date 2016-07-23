local vision = require 'torchnet-vision.env'

local image = {}
vision.image = image

image.transformimage = require 'torchnet-vision.image.transformimage'

return image
