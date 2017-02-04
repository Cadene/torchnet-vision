local argcheck = require 'argcheck'

local inceptionresnetv2 = {}

inceptionresnetv2.__download = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         os.execute('mkdir -p '..paths.dirname(filename)..';'
                  ..'wget http://webia.lip6.fr/~cadene/Downloads/inceptionresnetv2.t7'
                      ..' -O '..filename)
      end
}

inceptionresnetv2.load = argcheck{
   {name='filename', type='string', default='inceptionresnetv2.t7'},
   call =
      function(filename)
         if not path.exists(filename) then
            inceptionresnetv2.__download(filename)
         end
         return torch.load(filename)
      end
}

inceptionresnetv2.colorMode = 'RGB'
inceptionresnetv2.pixelRange = {0,1} -- [0,1] instead of [0,255]
inceptionresnetv2.inputSize = {3, 299, 299}
inceptionresnetv2.mean = torch.Tensor{0.5, 0.5, 0.5}
inceptionresnetv2.std  = torch.Tensor{0.5, 0.5, 0.5}

return inceptionresnetv2
