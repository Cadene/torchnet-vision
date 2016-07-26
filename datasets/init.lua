local vision = require 'torchnet-vision.env'

local datasets = {}
vision.datasets = datasets

datasets.utils       = require 'torchnet-vision.datasets.utils'
datasets.upmcfood101 = require 'torchnet-vision.datasets.upmcfood101'
datasets.mit67       = require 'torchnet-vision.datasets.mit67'

return datasets
