--------------------------
------Spiros Thermos------
---Visual Computing Lab---
--------------------------
---all rights reserved----

require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

paths.dofile('util.lua')

--Load pretrained model
if opt.mode=='train' then
  paths.dofile('model.lua') 
else
  model = torch.load(opt.loadModel)
  model:cuda()
end

print(opt)
--Select GPU, default=1
cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)
print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

if opt.mode=='test' then
	paths.dofile('val.lua')
elseif opt.mode=='features' then
	paths.dofile('featureExtraction.lua')
else
	paths.dofile('train.lua')
end

epoch = opt.epochNumber
if opt.mode=='test' then
	test()
elseif opt.mode=='features' then
	features()
else
  acclogger = optim.Logger(paths.concat(opt.save, 'accuracies.log'))
  for i=1,opt.nEpochs do
     train_acc = train()
     collectgarbage()
     --Plots
     acclogger:add{['train acc.'] = train_acc*100}
     acclogger:style{ ['train acc.'] = '+-'}
     acclogger:plot()
     epoch = epoch + 1
  end
end
