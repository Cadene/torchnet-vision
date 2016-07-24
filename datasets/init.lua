local vision = require 'torchnet-vision.env'

local datasets = {}
vision.datasets = datasets

datasets.utils       = require 'torchnet-vision.datasets.utils'
datasets.upmcfood101 = require 'torchnet-vision.datasets.upmcfood101'

return datasets
