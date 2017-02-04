local argcheck = require 'argcheck'

local inceptionv4 = {}

inceptionv4.__download = argcheck{
   {name='filename', type='string'},
   call =
      function(filename)
         os.execute('mkdir -p '..paths.dirname(filename)..';'
                  ..'wget http://webia.lip6.fr/~cadene/Downloads/inceptionv4.t7'
                      ..' -O '..filename)
      end
}

inceptionv4.load = argcheck{
   {name='filename', type='string', default='inceptionv4.t7'},
   call =
      function(filename)
         if not path.exists(filename) then
            inceptionv4.__download(filename)
         end
         return torch.load(filename)
      end
}

inceptionv4.colorMode = 'RGB'
inceptionv4.pixelRange = {0,1} -- [0,1] instead of [0,255]
inceptionv4.inputSize = {3, 299, 299}
inceptionv4.mean = torch.Tensor{0.5, 0.5, 0.5}
inceptionv4.std  = torch.Tensor{0.5, 0.5, 0.5}

return inceptionv4
