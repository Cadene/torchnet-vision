local argcheck = require 'argcheck'
local tnt = require 'torchnet'
local utils = require 'torchnet-vision.datasets.utils'
local lsplit = string.split

local upmcfood101 = {}

upmcfood101.__download = argcheck{
   {name='dirname', type='string'},
   call =
      function(dirname)
         os.execute('mkdir -p '..dirname..'; '
            --..'cp /net/big/cadene/doc/Deep6Framework/data/raw/UPMC_Food101/UPMC_Food101.tar.gz'..' '..dirname..'; '
            ..'wget http://visiir.lip6.fr/data/public/UPMC_Food101.tar.gz -P '..dirname..'; '
            ..'tar -xzf '..dirname..'/UPMC_Food101.tar.gz -C '..dirname)
      end
}

upmcfood101.load = argcheck{
   {name='dirname', type='string', default='data/raw/upmcfood101'},
   call =
      function(dirname)
         local dirimg   = paths.concat(dirname,'images')
         local dirtrain = paths.concat(dirimg,'train')
         local dirtest  = paths.concat(dirimg,'test')
         local traintxt = paths.concat(dirtrain,'TrainImages.txt')
         local testtxt  = paths.concat(dirtest,'TestImages.txt')
         if not paths.dirp(dirname) then
            upmcfood101.__download(dirname)
         end
         local classes, class2target = utils.findClasses(dirtrain)
         if not paths.filep(traintxt) then
            utils.findFilenames(dirtrain, classes, 'TrainImages.txt')
         end
         if not paths.filep(testtxt) then
            utils.findFilenames(dirtest, classes, 'TestImages.txt')
         end
         local loadSample = function(line)
               local spl = lsplit(line, '/')
               local sample  = {}
               sample.path   = line
               sample.label  = spl[#spl-1]
               sample.target = class2target[sample.label]
               return sample
            end
         local trainset = tnt.ListDataset{
            filename = traintxt,
            path     = dirtrain,
            load     = loadSample
         }
         local testset = tnt.ListDataset{
            filename = testtxt,
            path     = dirtest,
            load     = loadSample
         }
         return trainset, testset, classes, class2target
      end
}

return upmcfood101
