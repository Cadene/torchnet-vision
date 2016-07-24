local argcheck = require 'argcheck'

local vgg16 = {}

vgg16.__download = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         os.execute('mkdir -p '..paths.dirname(filename)..';'
                  ..'wget http://webia.lip6.fr/~cadene/Downloads/vgg16.t7'
                      ..' -O '..filename)
      end
}

vgg16.load = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         if not path.exists(filename) then
            vgg16.__download(filename)
         end
         return torch.load(filename)
      end
}

vgg16.loadFinetuning = argcheck{
   {name='filename', type='string'},
   {name='nclasses', type='number'},
   {name='ftfactor', type='number', default=10},
   {name='layerid',  type='number', default=38},
   call =
      function(filename, nclasses, ftfactor, layerid)
         local net = vgg16.load(filename)
         net:remove() -- nn.SoftMax
         net:remove() -- nn.Linear
         net:add(nn.Linear(2048, nclasses))
         for i=net:size(), layerid+1, -1 do
            net:get(i):reset()
         end
         net:insert(nn.GradientReversal(-1*ftfactor), layerid)
         return net
      end
}

vgg16.loadExtracting = argcheck{
   {name='filename', type='string'},
   {name='layerid',  type='number'},
   call =
      function(filename, layerid)
         local net = vgg16.load(filename)
         for i=net:size(), layerid+1, -1 do
            net:remove()
         end
         net:evaluate()
         return net
      end
}

vgg16.colorMode = 'BGR'
vgg16.inputSize = {3, 221, 221}
vgg16.mean = torch.Tensor{123.68, 116.779, 103.939}
-- no std

return vgg16
