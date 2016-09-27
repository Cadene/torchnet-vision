local argcheck = require 'argcheck'

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
   {name='dirname', type='string', default='data/raw/upmcfood101/images'},
   call =
      function(dirname)
         if not paths.dirp(dirname) then
            upmcfood101.__download(dirname)
         end
         local trainset, classes, class2target
            = utils.loadDataset(dirname..'/train')
         local testset, _, _
            = utils.loadDataset(dirname..'/test')
         return trainset, testset, classes, class2target
      end
}

return upmcfood101