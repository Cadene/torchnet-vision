local argcheck = require 'argcheck'

local inceptionv3 = {}

inceptionv3.__download = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         os.execute('mkdir -p '..paths.dirname(filename)..';'
                  ..'wget http://webia.lip6.fr/~cadene/Downloads/inceptionv3.t7'
                      ..' -O '..filename)
      end
}

inceptionv3.load = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         if not path.exists(filename) then
            inceptionv3.__download(filename)
         end
         return torch.load(filename)
      end
}

inceptionv3.loadFinetuning = argcheck{
   {name='filename', type='string'},
   {name='nclasses', type='number'},
   {name='ftfactor', type='number', default=10},
   call =
      function(filename, nclasses, ftfactor)
         local net = inceptionv3.load(filename)
         net:remove() -- nn.SoftMax
         net:remove() -- nn.Linear
         net:add(nn.GradientReversal(-1*ftfactor))
         net:add(nn.Linear(2048, nclasses))
         return net
      end
}

inceptionv3.loadExtracting = argcheck{
   {name='filename', type='string'},
   {name='layerid',  type='number'},
   call =
      function(filename, layerid)
         local net = inceptionv3.load(filename)
         for i=net:size(), layerid+1, -1 do
            net:remove()
         end
         net:evaluate()
         return net
      end
}

inceptionv3.colorMode = 'RGB'
inceptionv3.inputSize = {3, 299, 299}
inceptionv3.mean = torch.Tensor{128, 128, 128}
inceptionv3.std  = torch.Tensor{128, 128, 128}

return inceptionv3
