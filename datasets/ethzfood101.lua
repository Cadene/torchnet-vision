local argcheck = require 'argcheck'

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
         if not paths.dirp(dirname) then
            ethzfood101.__download(dirname)
         end
         return trainset, testset, classes, class2target
      end
}

return ethzfood101