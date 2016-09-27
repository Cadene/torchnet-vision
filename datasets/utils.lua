local argcheck = require 'argcheck'
local tnt = require 'torchnet'

local utils = {}

utils.findClasses = argcheck{
   {name='dirname', type='string'},
   call =
      function(dirname)
         local find = 'find'
         local handle = io.popen(find..' '..dirname..' -mindepth 1 -maxdepth 1 -type d'
                                 ..' | grep -o \'[^/]*$\' | sort')
         local classes = {}
         local class2target = {}
         local key = 1
         for class in handle:lines() do
            table.insert(classes, class)
            class2target[class] = key
            key = key + 1
         end
         handle:close()
         return classes, class2target
      end
}

utils.findFilenames = argcheck{
   {name='dirname', type='string'},
   {name='classes', type='table'},
   {name='filename', type='string', default='filename.txt'},
   call =
      function(dirname, classes, filename)
         local pathfilename = dirname..'/'..filename
         local find = 'find'
         local extensionList = {'jpg', 'png', 'JPG', 'PNG', 'JPEG',
                                'ppm', 'PPM', 'bmp', 'BMP'}
         local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
         for i = 2, #extensionList do
            findOptions = findOptions .. ' -o'
                              ..' -iname "*.' .. extensionList[i] .. '"'
         end
         assert(not paths.filep(pathfilename),
            'filename already exists, you should remove it first')
         for _, class in pairs(classes) do
            os.execute(find..' "'..dirname..'/'..class..'" '..findOptions
                       ..' | grep -o \'[^/]*/[^/]*$\' >> '..pathfilename)
         end
         return pathfilename
      end
}

utils.loadDataset = argcheck{
   {name='dirname', type='string'},
   {name='filename', type='string', default='filename.txt'},
   call =
      function(dirname, filename)
         local classes, class2target = utils.findClasses(dirname)
         local pathfilename = dirname..'/'..filename
         if not paths.filep(pathfilename) then
            utils.findFilenames(dirname, classes, filename)
         end
         local dataset = tnt.ListDataset{
            filename = pathfilename,
            path = dirname,
            load = function(line)
               local sample = {
                  path = line
               }
               return sample
            end
         }
         -- return {
         --    dataset      = dataset,
         --    classes      = classes,
         --    class2target = class2target
         -- }
         return dataset, classes, class2target
      end
}

return utils