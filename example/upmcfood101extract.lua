local tnt = require 'torchnet'
local vision = require 'torchnet-vision'
require 'image'
require 'os'
require 'optim'
ffi = require 'ffi'
local logtext   = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'
local transformimage = require 'torchnet-vision.image.transformimage'
local upmcfood101 = require 'torchnet-vision.datasets.upmcfood101'

local cmd = torch.CmdLine()
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 25, 'batch size')
cmd:option('-nthread', 3, 'threads number for parallel iterator')
cmd:option('-model', 'vgg16', 'options: vgg16 | vggm | resnet200 | inceptionv3')
cmd:option('-layerid', 37, 'ex: vgg16 + 37 = 2nd FC layer after ReLU ')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.date  = os.date("%y_%m_%d_%X")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local pathdataset  = paths.concat('example/data/processed/upmcfood101')
local pathtrainset = paths.concat(pathdataset,'trainset.t7')
local pathtestset  = paths.concat(pathdataset,'testset.t7')
local pathdata     = paths.concat('example/data/raw/upmcfood101')
local pathmodel    = paths.concat('example/models',config.model,'net.t7')
local pathextract  = paths.concat('example/features/upmcfood101',config.date)
local pathconfig   = paths.concat(pathextract,'config.t7')

local model = vision.models[config.model]
local net = model.loadExtracting{
   filename = pathmodel,
   layerid  = config.layerid
}
print(net)
local criterion = nn.CrossEntropyCriterion():float()

local trainset, testset, classes, class2target = upmcfood101.load()
-- testset  = testset:shuffle(300)
-- trainset = trainset:shuffle(300)

local function addTransforms(dataset, model)
   dataset = dataset:transform(function(sample)
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         vision.image.transformimage.scale(model.inputSize[2]),
         vision.image.transformimage.centerCrop(model.inputSize[2]),
         vision.image.transformimage.colorNormalize{
            mean = model.mean,
            std  = model.std
         }
      }(sample.path)
      return sample
   end)
   return dataset
end
testset  = addTransforms(testset, model)
trainset = addTransforms(trainset, model)
function trainset:manualSeed(seed) torch.manualSeed(seed) end

os.execute('mkdir -p '..pathdataset)
os.execute('mkdir -p '..pathextract)
torch.save(pathconfig, config)
torch.save(pathtrainset, trainset)
torch.save(pathtestset, testset)

local function getIterator(mode)
   local iterator = tnt.ParallelDatasetIterator{
      nthread   = config.nthread,
      init      = function()
         require 'torchnet'
         require 'torchnet-vision'
      end,
      closure   = function(threadid)
         local dataset = torch.load(pathdataset..'/'..mode..'set.t7')
         return dataset:batch(config.bsize)
      end,
      transform = function(sample)
         sample.target = torch.Tensor(sample.target):view(-1,1)
         return sample
      end
   }
   print('Stats of '..mode..'set')
   for i, v in pairs(iterator:exec('size')) do
      print(i, v)
   end
   return iterator
end

local meter = {
   timem = tnt.TimeMeter{unit = false},
}

local engine = tnt.OptimEngine()
local file
local count, nbatch
engine.hooks.onStart = function(state)
   count = 1
   nbatch = state.iterator:execSingle("size")
   for _,m in pairs(meter) do m:reset() end
   print(engine.mode)
   file = assert(io.open(pathextract..'/'..engine.mode..'set.csv', "w"))
   file:write('path;gttarget;gtclass')
   for i=1, #classes do file:write(';pred'..i) end
   file:write("\n")
end
engine.hooks.onForward = function(state)
   local output = state.network.output
   print('Mode: '..engine.mode,
         'Inputid: '..count..' / '..nbatch,
         'Size: '..state.sample.input:size(1)
            ..' '..output:size(1)
            ..' '..state.sample.target:size(1)
            ..' '..#state.sample.path)
   count = count + 1
   if state.sample.input:size(1) == output:size(1) then -- hotfix
      for i=1, output:size(1) do
         file:write(state.sample.path[i]);
         if engine.mode ~= 'test' then
            file:write(';')
            file:write(state.sample.target[i][1]); file:write(';')
            file:write(state.sample.label[i])
         end
         for j=1, output:size(2) do
            file:write(';'); file:write(output[i][j])
         end
         file:write("\n")
      end
   end
end
engine.hooks.onEnd = function(state)
   print('End of extracting on '..engine.mode..'set')
   print('Took '..meter.timem:value())
   file:close()
end

if config.usegpu then
   require 'cutorch'
   cutorch.manualSeed(config.seed)
   require 'cunn'
   require 'cudnn'
   cudnn.convert(net, cudnn)
   net       = net:cuda()
   criterion = criterion:cuda()
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

print('Extracting trainset ...')
engine.mode = 'train'
engine:test{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion
}

print('Extracting testset ...')
engine.mode = 'test'
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion
}
