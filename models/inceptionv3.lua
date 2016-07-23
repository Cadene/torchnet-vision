local argcheck = require 'argcheck'

local inceptionv3 = {}

inceptionv3.__download = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         os.execute('mkdir -p '..paths.dirname(filename)..';'
                  ..'wget http://webia.lip6.fr/~cadene/Downloads/InceptionV3.t7'
                      ..' -O '..filename)
      end
}

inceptionv3.load = argcheck{
   {name='filename', type='string'},
   {name='ftfactor', type='number', opt=true},
   {name='nclasses', type='number', opt=true},
   call =
      function(filename, ftfactor, nclasses)
         if not path.exists(filename) then
            inceptionv3.__download(filename)
         end
         local net = torch.load(filename)
         if ftfactor then
            net:remove() -- nn.SoftMax
            net:remove() -- nn.Linear
            net:add(nn.GradientReversal(-1*ftfactor))
            net:add(nn.Linear(2048, nclasses))
         end
         return net:float()
      end
}

inceptionv3.load = argcheck{
   {name='filename', type='string'},
   {name='fextract', type='boolean', opt=true},
   overload = inceptionv3.load,
   force = true,
   call =
      function(filename, fextract)
         if not path.exists(filename) then
            inceptionv3.__download(filename)
         end
         local net = torch.load(filename)
         if fextract then
            net:remove() -- nn.SoftMax
            net:remove() -- nn.Linear
         end
         return net:float()
      end
}

inceptionv3.mean = function()
   return torch.Tensor{128, 128, 128}
end

inceptionv3.std = function()
   return torch.Tensor{128, 128, 128}
end

return inceptionv3
