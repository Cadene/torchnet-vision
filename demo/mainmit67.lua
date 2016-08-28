require 'image'
require 'os'
require 'optim'
ffi = require 'ffi'
local tnt            = require 'torchnet'
local vision         = require 'torchnet-vision'
local logtext        = require 'torchnet.log.view.text'
local logstatus      = require 'torchnet.log.view.status'
local transformimage = require 'torchnet-vision.image.transformimage'
local lsplit = string.split -- string can't be serialized thus we use a local var

local cmd = torch.CmdLine()
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-usecudnn', true, 'use cudnn')
cmd:option('-bsize', 20, 'batch size')
cmd:option('-nepoch', 50, 'epoch number')
cmd:option('-optim', 'adam', 'optimization method, options: sgd | ...')
cmd:option('-lr', 1e-4, 'learning rate')
cmd:option('-lrd', 0, 'learning rate decay (adam compatible)')
cmd:option('-ftfactor', 10, 'fine tuning factor')
cmd:option('-nthread', 3, 'threads number for parallel iterator')
cmd:option('-model', 'vgg16', 'model name, options: inceptionv3 | vggm')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.date  = os.date("%y_%m_%d_%X")
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path          = './demo'
local pathmodel     = path..'/models/'..config.model..'/net.t7'
local pathdata      = path..'/data/raw/mit67'
local pathdataset   = path..'/data/processed/mit67'
local pathlog       = path..'/logs/mit67/'..config.date
local pathtrainset  = pathdataset..'/trainset.t7'
local pathtestset   = pathdataset..'/testset.t7'
local pathtrainlog  = pathlog..'/trainlog.txt'
local pathtestlog   = pathlog..'/testlog.txt'
local pathbestepoch = pathlog..'/bestepoch.t7'
local pathbestnet   = pathlog..'/net.t7'
local pathconfig    = pathlog..'/config.t7'
os.execute('mkdir -p '..pathdataset) -- here we save datasets for threads
os.execute('mkdir -p '..pathlog)     -- here we save experiments logs and best net
torch.save(pathconfig, config)

local trainset, testset, classes, class2target
   = vision.datasets.mit67.load(pathdata)
print('Trainset size', trainset:size())
print('Testset size',  testset:size())
print('Classes number', #classes)
print('First class and its associated target', classes[1], class2target[classes[1]])
print('')

local model = vision.models[config.model]
local net = model.loadFinetuning{ -- download included
   filename = pathmodel,
   ftfactor = config.ftfactor,
   nclasses = #classes
}
print(net)
print('Input size', {model.inputSize[1],model.inputSize[2],model.inputSize[3]})
print('Color mode', model.colorMode)
print('')
local criterion = nn.CrossEntropyCriterion():float()

local function addTransforms(dataset, model)
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.path,'/')
      sample.label  = spl[#spl-1]
      sample.target = class2target[sample.label]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         transformimage.randomScale{
            minSize = model.inputSize[2], -- 224 for vgg16 or 299 for inceptionv3
            maxSize = model.inputSize[2] + 20   -- randomly load bigger image
         }, -- keep image ratio by cropping instead of rescaling to a squared image
         transformimage.randomCrop(model.inputSize[2]),
         transformimage.verticalFlip(0.5),
         transformimage.rotation(0.05),
         function(img)
            if model.colorMode == 'BGR' then
               return transformimage.moveColor()(img * 255)
            else          -- vggm and vgg16 take img color=BGR intensity=[0,255]
               return img -- inceptionv3   takes img color=RGB intensity=[0,1]
            end
         end,
         transformimage.colorNormalize(model.mean, model.std)
      }(sample.path)
      return sample
   end)
   return dataset
end

trainset = trainset:shuffle() -- trainset:shuffle(300) to try out with 300 images
trainset = addTransforms(trainset, model)
testset  = addTransforms(testset, model)
-- manualSeed is called after each epoch before shuffling the trainset
function trainset:manualSeed(seed) torch.manualSeed(seed) end
torch.save(pathtrainset, trainset)
torch.save(pathtestset, testset)

local function getIterator(mode) -- mode options= train | test
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
   -- all threads have the same # of batch
   for i, v in pairs(iterator:exec('size')) do
      print('Theadid='..i, 'Batch number='..v)
   end
   return iterator
end

local meter = {
   avgvm = tnt.AverageValueMeter(),
   confm = tnt.ConfusionMeter{k=#classes, normalized=true},
   timem = tnt.TimeMeter{unit = false},
   clerr = tnt.ClassErrorMeter{topk = {1,5}}
}

local function createLog(mode, pathlog)
   local keys = {'epoch', 'loss', 'acc1', 'acc5', 'time'}
   local format = {'%d', '%.5f', '%3.2f%%', '%3.2f%%', '%.1f'}
   for i=1, #keys do format[i] = keys[i]..' '..format[i] end
   local log = tnt.Log{
      keys = keys,
      onFlush = {
         logtext{filename=pathlog, keys=keys},
         logtext{keys=keys, format=format},
      },
      onSet = {
         logstatus{filename=pathlog},
         logstatus{}, -- print status to screen
      }
   }
   log:status("Mode "..mode)
   return log
end
local log = {
   train = createLog('train', pathtrainlog),
   test  = createLog('test', pathtestlog)
}

local engine = tnt.OptimEngine()
engine.hooks.onStart = function(state)
   for _, m in pairs(meter) do m:reset() end
end
engine.hooks.onStartEpoch = function(state) -- training only
   engine.epoch = engine.epoch and (engine.epoch + 1) or 1
end
engine.hooks.onForwardCriterion = function(state)
   meter.timem:incUnit()
   meter.avgvm:add(state.criterion.output)
   meter.clerr:add(state.network.output, state.sample.target)
   meter.confm:add(state.network.output, state.sample.target)
   log[engine.mode]:set{
      epoch = engine.epoch,
      loss  = meter.avgvm:value(),
      acc1  = 100 - meter.clerr:value{k = 1},
      acc5  = 100 - meter.clerr:value{k = 5},
      time  = meter.timem:value()
   }
   print(string.format('%s epoch: %i; avg. loss: %2.4f; avg. acctop1: %2.4f%%',
      engine.mode, engine.epoch, meter.avgvm:value(), 100 - meter.clerr:value{k = 1}))
end
engine.hooks.onEnd = function(state)
   print('End of epoch '..engine.epoch..' on '..engine.mode..'set')
   log[engine.mode]:flush()
   print('Confusion matrix saved (rows = gt, cols = pred)\n')
   image.save(pathlog..'/confm_epoch,'..engine.epoch..'.pgm', meter.confm:value())
end
if config.usegpu then
   require 'cutorch'
   cutorch.manualSeed(config.seed)
   require 'cunn'
   if config.usecudnn then
      require 'cudnn'
      cudnn.convert(net, cudnn)
   end
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

-- Iterator
local trainiter = getIterator('train')
local testiter  = getIterator('test')

local bestepoch = {
   acctop1 = 0,
   acctop5 = 0,
   epoch   = 0
}

for epoch = 1, config.nepoch do
   print('Training ...')
   engine.mode = 'train'
   trainiter:exec('manualSeed', config.seed + epoch) -- call trainset:manualSeed(seed)
   trainiter:exec('resample') -- shuffle trainset
   engine:train{
      maxepoch    = 1, -- we control the epoch with for loop
      network     = net,
      iterator    = trainiter,
      criterion   = criterion,
      optimMethod = optim[config.optim],
      config      = {
         learningRate      = config.lr,
         learningRateDecay = config.lrd
      },
   }
   print('Testing ...')
   engine.mode = 'test'
   engine:test{
      network   = net,
      iterator  = testiter,
      criterion = criterion,
   }
   if bestepoch.acctop1 < 100 - meter.clerr:value{k = 1} then
      bestepoch = {
         acctop1 = 100 - meter.clerr:value{k = 1},
         acctop5 = 100 - meter.clerr:value{k = 5},
         epoch = epoch,
         confm = meter.confm:value():clone()
      }
      torch.save(pathbestepoch, bestepoch)
      torch.save(pathbestnet, net:clearState())
   end
end
