require 'torch'

local tnt = require 'torchnet'
local vision = require 'torchnet-vision.env'
local doc = require 'argcheck.doc'

-- doc[[]]

require 'torchnet-vision.image'

require 'torchnet-vision.models'

require 'torchnet-vision.test.test'

tnt.makepackageserializable(vision, 'torchnet-vision')

return vision
