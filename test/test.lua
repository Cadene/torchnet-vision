local __main__ = package.loaded['torchnet-vision.env'] == nil

local vision = require 'torchnet-vision.env'
local tds = require 'tds'

if __main__ then
   require 'torchnet-vision'
end

local tester = torch.Tester()
tester:add(paths.dofile('transformimage.lua')(tester))

function vision.test(tests)
   tester:run(tests)
   return tester
end

if __main__ then
   require 'torchnet-vision'
   if #arg > 0 then
      vision.test(arg)
   else
      vision.test()
   end
end
