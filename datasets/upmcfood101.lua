local argcheck = require 'argcheck'
local utils = require 'torchnet-vision.datasets.utils'

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
         local dirimg   = paths.concat(dirname, 'images')
         local traintxt = paths.concat(dirname, 'TrainImages.txt')
         local testtxt  = paths.concat(dirname, 'TestImages.txt')
         if not paths.dirp(dirname) then
            upmcfood101.__download(dirname)
         end
         local classes, class2target = utils.findClasses(dirimg)
         if not paths.filep(traintxt) then
            utils.findFilenames(dirimg..'/train', classes, traintxt)
         end
         if not paths.filep(testtxt) then
            utils.findFilenames(dirimg..'/test', classes, testtxt)
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

return upmcfood101