local vision = require 'torchnet-vision.env'

local datasets = {}
vision.datasets = datasets

datasets.upmcfood101 = require 'torchnet-vision.datasets.upmcfood101'

return datasets
