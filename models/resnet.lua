local argcheck = require 'argcheck'

local resnet = {}

resnet.__download = argcheck{
   {name='filename', type='string'},
   {name='length', type='number', default=200, help='18, 34, 50, 101, 152, 200',
    check=function(length)
            return length == 18  or length == 34  or length == 50 or
                   length == 101 or length == 152 or length == 200
          end},
   call =
      function(filename, length)
         os.execute('mkdir -p '..paths.dirname(filename)..';'
                  ..'wget https://d2j0dndfm35trm.cloudfront.net/resnet-'..length..'.t7'
                      ..' -O '..filename)
      end
}

resnet.load = argcheck{
   {name='filename', type='string'},
   {name='length', type='number', default=200},
   call =
      function(filename)
         if not path.exists(filename) then
            resnet.__download(filename)
         end
         return torch.load(filename)
      end
}

resnet.colorMode = 'RGB'
resnet.inputSize = {3, 224, 224}
resnet.mean = torch.Tensor{0.485, 0.456, 0.406}
resnet.std  = torch.Tensor{0.229, 0.224, 0.225}

return resnet
