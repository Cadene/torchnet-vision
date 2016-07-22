--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

local vision = require 'torchnet-vision.env'
local argcheck = require 'argcheck'

require 'image'

local TransformImage =
   torch.class('vision.TransformImage', vision)

TransformImage.__init = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='opt', type='table', opt=true},
   call = 
      function(self, opt)
         self.__opt = opt
      end
}

TransformImage.colorNormalize = argcheck{
   doc = [[]],
   {name='self', type='vision.TransformImage'},
   {name='mean', type='torch.*Tensor', opt=true},
   {name='std', type='torch.*Tensor', opt=true},
   call =
      function(self, mean, std)
         return function(img)
            if not (mean or std) then
               return img
            end
            img = img:clone()
            for i=1,3 do
               img[i]:add(-mean[i])
               if std then
                  img[i]:div(std[i])
               end
            end
            return img
         end
      end
}

-- Scales the smaller edge to size
TransformImage.scale = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='size', type='number'},
   {name='interpolation', type='string', default='bicubic'},
   call =
      function(self, size, interpolation)
         return function(img)
            local w, h = img:size(3), img:size(2)
            if (w <= h and w == size) or (h <= w and h == size) then
               return img
            end
            if w < h then
               return image.scale(img, size, h/w * size, interpolation)
            else
               return image.scale(img, w/h * size, size, interpolation)
            end
         end
      end
}

-- Crop to centered rectangle
TransformImage.centerCrop = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='size', type='number'},
   call =
      function(self, size)
         return function(img)
            local w1 = math.ceil((img:size(3) - size)/2)
            local h1 = math.ceil((img:size(2) - size)/2)
            return image.crop(img, w1, h1, w1 + size, h1 + size) -- center patch
         end
      end
}

-- Random crop form larger image with optional zero padding
TransformImage.randomCrop = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='size', type='number'},
   {name='padding', type='number', default=0},
   call =
      function(self, size, padding)
         return function(img)
            if padding > 0 then
               local temp = img.new(3, img:size(2) + 2*padding, img:size(3) + 2*padding)
               temp:zero()
                  :narrow(2, padding+1, img:size(2))
                  :narrow(3, padding+1, img:size(3))
                  :copy(img)
               img = temp
            end

            local w, h = img:size(3), img:size(2)
            if w == size and h == size then
               return img
            end

            local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
            local out = image.crop(img, x1, y1, x1 + size, y1 + size)
            assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
            return out
         end
      end
}

-- Four corner patches and center crop from image and its horizontal reflection
TransformImage.tenCrop = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='size', type='number'},
   call =
      function(self, size)
         local centerCrop = TransformImage.CenterCrop(size)

         return function(img)
            local w, h = img:size(3), img:size(2)

            local output = {}
            for _, img in ipairs{img, image.hflip(img)} do
               table.insert(output, centerCrop(img))
               table.insert(output, image.crop(img, 0, 0, size, size))
               table.insert(output, image.crop(img, w-size, 0, w, size))
               table.insert(output, image.crop(img, 0, h-size, size, h))
               table.insert(output, image.crop(img, w-size, h-size, w, h))
            end

            -- View as mini-batch
            for i, img in ipairs(output) do
               output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
            end

            return img.cat(output, 1)
         end
      end
}

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
TransformImage.randomScale = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='minSize', type='number'},
   {name='maxSize', type='number'},
   call =
      function(self, minSize, maxSize)
         return function(img)
            local w, h = img:size(3), img:size(2)

            local targetSz = torch.random(minSize, maxSize)
            local targetW, targetH = targetSz, targetSz
            if w < h then
               targetH = torch.round(h / w * targetW)
            else
               targetW = torch.round(w / h * targetH)
            end

            return image.scale(img, targetW, targetH, 'bicubic')
         end
      end
}

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
TransformImage.randomSizedCrop = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='size', type='number'},
   call =
      function(self, size)
         local scale = TransformImage.Scale(size)
         local crop = TransformImage.CenterCrop(size)

         return function(img)
            local attempt = 0
            repeat
               local area = img:size(2) * img:size(3)
               local targetArea = torch.uniform(0.08, 1.0) * area

               local aspectRatio = torch.uniform(3/4, 4/3)
               local w = torch.round(math.sqrt(targetArea * aspectRatio))
               local h = torch.round(math.sqrt(targetArea / aspectRatio))

               if torch.uniform() < 0.5 then
                  w, h = h, w
               end

               if h <= img:size(2) and w <= img:size(3) then
                  local y1 = torch.random(0, img:size(2) - h)
                  local x1 = torch.random(0, img:size(3) - w)

                  local out = image.crop(img, x1, y1, x1 + w, y1 + h)
                  assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

                  return image.scale(out, size, size, 'bicubic')
               end
               attempt = attempt + 1
            until attempt >= 10

            -- fallback
            return crop(scale(img))
         end
      end
}

TransformImage.horizontalFlip = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='prob', type='number', default=0.5},
   call =
      function(self, prob)
         return function(img)
            if torch.uniform() < prob then
               img = image.hflip(img)
            end
            return img
         end
      end
}

TransformImage.verticalFlip = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='prob', type='number', default=0.5},
   call =
      function(self, prob)
         return function(img)
            if torch.uniform() < prob then
               img = image.vflip(img)
            end
            return img
         end
      end
}

TransformImage.rotation = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='deg', type='number'},
   call =
      function(self, deg)
         return function(img)
            if deg ~= 0 then
               img = image.rotate(img, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
            end
            return img
         end
      end
}

-- Lighting noise (AlexNet-style PCA-based noise)
TransformImage.lighting = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='alphastd', type='number'},
   {name='eigval', type='torch.*Tensor',
     default=torch.Tensor{ 0.2175, 0.0188, 0.0045 },
     check=function(x)
              return x:dim() == 1 and x:size(1) == 3
           end},
   {name='eigvec', type='torch.*Tensor',
     default=torch.Tensor{
       { -0.5675,  0.7192,  0.4009 },
       { -0.5808, -0.0045, -0.8140 },
       { -0.5836, -0.6948,  0.4203 },
     },
     check=function(x)
              return x:dim() == 2 and x:size(1) == 3
                                  and x:size(2) == 3
           end},
   call =
      function(self, alphastd, eigval, eigvec)
         return function(img)
            if alphastd == 0 then
               return img
            end

            local alpha = torch.Tensor(3):normal(0, alphastd)
            local rgb = eigvec:clone()
               :cmul(alpha:view(1, 3):expand(3, 3))
               :cmul(eigval:view(1, 3):expand(3, 3))
               :sum(2)
               :squeeze()

            img = img:clone()
            for i=1,3 do
               img[i]:add(rgb[i])
            end
            return img
         end
      end
}

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

TransformImage.grayscale = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='rgbval', type='torch.*Tensor',
     default=torch.Tensor{ 0.299, 0.587, 0.114 },
     check=function(x)
              return x:dim() == 1 and x:size(1) == 3
           end},
   call =
      function(self, rgbval)
         return function(img)
            local dst = torch.new():resizeAs(img)
            dst[1]:zero()
            dst[1]:add(rgbval[1], img[1])
                  :add(rgbval[2], img[2])
                  :add(rgbval[3], img[3])
            dst[2]:copy(dst[1])
            dst[3]:copy(dst[1])
            return dst
         end
      end
}

TransformImage.moveColor = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='colormap', type='torch.ByteTensor',
     default=torch.ByteTensor{ 3, 2, 1 },
     check=function(x)
              return x:dim() == 1 and x:size(1) == 3
           end},
   call =
      function(self)
         return function(img)
            local dst = torch.new():resizeAs(img)
            dst[colormap[1]]:copy(img[1])
            dst[colormap[2]]:copy(img[2])
            dst[colormap[3]]:copy(img[3])
            return dst
         end
      end
}

TransformImage.saturation = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='var', type='number'},
   call =
      function(self, var)
         local gs

         return function(img)
            gs = gs or img.new()
            self:grayscale(gs, img)

            local alpha = 1.0 + torch.uniform(-var, var)
            blend(img, gs, alpha)
            return img
         end
      end
}

TransformImage.brightness = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='var', type='number'},
   call =
      function(self, var)
         local gs

         return function(img)
            gs = gs or img.new()
            gs:resizeAs(img):zero()

            local alpha = 1.0 + torch.uniform(-var, var)
            blend(img, gs, alpha)
            return img
         end
      end
}

TransformImage.contrast = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='var', type='number'},
   call =
      function(self, var)
         local gs

         return function(img)
            gs = gs or img.new()
            self:grayscale(gs, img)
            gs:fill(gs[1]:mean())

            local alpha = 1.0 + torch.uniform(-var, var)
            blend(img, gs, alpha)
            return img
         end
      end
}

TransformImage.randomOrder = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='ts', type='table'},
   call =
      function(self, ts)
         return function(img)
            local img = img.img or img
            local order = torch.randperm(#ts)
            for i=1,#ts do
               img = ts[order[i]](img)
            end
            return img
         end
      end
}

TransformImage.colorJitter = argcheck{
   {name='self', type='vision.TransformImage'},
   {name='brightness', type='number', default=0},
   {name='contrast', type='number', default=0},
   {name='saturation', type='number', default=0},
   call =
      function(self, brightness, contrast, saturation)
         local ts = {}
         if brightness ~= 0 then
            table.insert(ts, self:brightness(brightness))
         end
         if contrast ~= 0 then
            table.insert(ts, self:contrast(contrast))
         end
         if saturation ~= 0 then
            table.insert(ts, self:saturation(saturation))
         end

         if #ts == 0 then
            return function(img) return img end
         end

         return self:randomOrder(ts)
      end
}

return TransformImage
