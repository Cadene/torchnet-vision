local argcheck = require 'argcheck'

local vggm = {}

vggm.__download = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         os.execute('mkdir -p '..paths.dirname(filename)..';'
                  ..'wget http://webia.lip6.fr/~cadene/Downloads/vggm.t7'
                      ..' -O '..filename)
      end
}

vggm.load = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         if not path.exists(filename) then
            vggm.__download(filename)
         end
         return torch.load(filename)
      end
}

vggm.loadFinetuning = argcheck{
   {name='filename', type='string'},
   {name='nclasses', type='number'},
   {name='ftfactor', type='number', default=10},
   {name='layerid',  type='number', default=22},
   call =
      function(filename, nclasses, ftfactor, layerid)
         local net = vggm.load(filename)
         net:remove(24) -- nn.SoftMax
         net:remove(23) -- nn.Linear
         net:add(nn.Linear(4096, nclasses))
         for i=net:size(), layerid+1, -1 do
            net:get(i):reset()
         end
         net:insert(nn.GradientReversal(-1*ftfactor), layerid)
         return net
      end
}

vggm.loadExtracting = argcheck{
   {name='filename', type='string'},
   {name='layerid',  type='number'},
   call =
      function(filename, layerid)
         local net = vggm.load(filename)
         for i=net:size(), layerid+1, -1 do
            net:remove()
         end
         net:evaluate()
         return net
      end
}

vggm.colorMode = 'BGR'
vggm.inputSize = {3, 221, 221}
vggm.mean = torch.Tensor{123.68, 116.779, 103.939}
-- no std

return vggm
