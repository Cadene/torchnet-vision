local argcheck = require 'argcheck'

local tnt   = require 'torchnet'
local utils = require 'torchnet-vision.datasets.utils'

local mit67 = {}

mit67.__download = argcheck{
   {name='dirname', type='string', default='data/raw/mit67'},
   call =
      function(dirname)
         local urlremote = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
         local urltrainimages = 'http://web.mit.edu/torralba/www/TrainImages.txt'
         local urltestimages = 'http://web.mit.edu/torralba/www/TestImages.txt'
         os.execute('mkdir -p '..dirname..'; '..
           'wget '..urlremote..' -P '..dirname..'; '..
           'tar -C '..dirname..' -xf '..dirname..'/indoorCVPR_09.tar')
         os.execute('wget '..urltrainimages..' -P '..dirname)
         os.execute('wget '..urltestimages..' -P '..dirname)
      end
}

mit67.load = argcheck{
   {name='dirname', type='string', default='data/raw/mit67'},
   call =
      function(dirname)
         local dirimg   = paths.concat(dirname, 'Images')
         local traintxt = paths.concat(dirname, 'TrainImages.txt')
         local testtxt  = paths.concat(dirname, 'TestImages.txt')
         if not (paths.dirp(dirname)   and paths.dirp(dirimg) and
                 paths.filep(traintxt) and paths.filep(testtxt)) then
            mit67.__download(dirname)
         end
         local classes, class2target = utils.findClasses(dirimg)
         local trainset = tnt.ListDataset{
            filename = traintxt,
            path = dirimg,
            load = function(line)
               local sample = {path=line}
               return sample
            end
         }
         local testset = tnt.ListDataset{
            filename = testtxt,
            path = dirimg,
            load = function(line)
               local sample = {path=line}
               return sample
            end
         }
         return trainset, testset, classes, class2target
      end
}

return mit67



