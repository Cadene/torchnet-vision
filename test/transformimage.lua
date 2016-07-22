local vision = require 'torchnet-vision.env'
local tds = require 'tds'

local tester
local test = torch.TestSuite()

function test.TransformImage()
end

return function(_tester_)
   tester = _tester_
   return test
end