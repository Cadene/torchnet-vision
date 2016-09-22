local argcheck = require 'argcheck'

local overfeat = {}

overfeat.__download = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         os.execute('mkdir -p '..paths.dirname(filename)..';'
                  ..'wget http://webia.lip6.fr/~cadene/Downloads/overfeat.t7'
                      ..' -O '..filename)
      end
}

overfeat.load = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         if not path.exists(filename) then
            overfeat.__download(filename)
         end
         return torch.load(filename)
      end
}

-- overfeat.loadFinetuning = argcheck{
--    {name='filename', type='string'},
--    {name='nclasses', type='number'},
--    {name='ftfactor', type='number', default=10},
--    {name='layerid',  type='number', default=38},
--    call =
--       function(filename, nclasses, ftfactor, layerid)
--          local net = overfeat.load(filename)
--          net:remove() -- nn.SoftMax
--          net:remove() -- nn.Linear
--          net:add(nn.Linear(4096, nclasses))
--          for i=net:size(), layerid+1, -1 do
--             net:get(i):reset()
--          end
--          net:insert(nn.GradientReversal(-1.0/ftfactor), layerid)
--          return net
--       end
-- }

-- overfeat.loadExtracting = argcheck{
--    {name='filename', type='string'},
--    {name='layerid',  type='number'},
--    call =
--       function(filename, layerid)
--          local net = overfeat.load(filename)
--          for i=net:size(), layerid+1, -1 do
--             net:remove()
--          end
--          net:evaluate()
--          return net
--       end
-- }

overfeat.colorMode = 'RGB'
overfeat.inputSize = {3, 224, 224}
overfeat.mean = torch.Tensor{118.380948, 118.380948, 118.380948}
overfeat.std  = torch.Tensor{61.896913, 61.896913, 61.896913}

return overfeat
