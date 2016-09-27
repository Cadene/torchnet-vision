local argcheck = require 'argcheck'
local tnt = require 'torchnet'
local utils = require 'torchnet-vision.datasets.utils'
local lsplit = string.split

local ethzfood101 = {}

ethzfood101.__download = argcheck{
   {name='dirname', type='string'},
   call =
      function(dirname)
         os.execute('mkdir -p '..dirname..'; '
            --..'cp /net/big/cadene/doc/Deep6Framework/data/raw/UPMC_Food101/UPMC_Food101.tar.gz'..' '..dirname..'; '
            ..'wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz -P '..dirname..'; '
            ..'tar -xzf '..dirname..'/food-101.tar.gz -C '..dirname)
      end
}

ethzfood101.load = argcheck{
   {name='dirname', type='string', default='data/raw/ethzfood101'},
   call =
      function(dirname)
         local dirimg   = paths.concat(dirname,'food-101','images')
         local traintxt = paths.concat(dirname,'food-101','meta','train.txt')
         local testtxt  = paths.concat(dirname,'food-101','meta','test.txt')
         if not paths.dirp(dirname) then
            ethzfood101.__download(dirname)
         end
         local classes, class2target = utils.findClasses(dirimg)
         local loadSample = function(line)
            local spl = lsplit(line, '/')
            local sample  = {}
            sample.path   = line..'.jpg'
            sample.label  = spl[#spl-1]
            sample.target = class2target[sample.label]
            return sample
         end
         local trainset = tnt.ListDataset{
            filename = traintxt,
            path     = dirimg,
            load     = loadSample
         }
         local testset = tnt.ListDataset{
            filename = testtxt,
            path     = dirimg,
            load     = loadSample
         }
         return trainset, testset, classes, class2target
      end
}

return ethzfood101
