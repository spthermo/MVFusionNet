--------------------------
------Spiros Thermos------
---Visual Computing Lab---
--------------------------
---all rights reserved----

local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch Shrec Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', './results/', 'subdirectory in which to save/log experiments')
    cmd:option('-inputDataPath', 'data', 'Home of ImageNet dataset')    
    cmd:option('-loadModel', 'models/cnn1_30.t7', 'Load a pre-trained model from directory')
    cmd:option('-dirName', 'train', 'Name of the directory with the set to train/test')
    cmd:option('-targetDirName', 't7/test', 'Name of the directory to save the extracted features')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',      'cudnn', 'Options: cudnn | cunn')
    ------------- Data options ------------------------
    cmd:option('-imageSize',         224,    'width/height of input')
    cmd:option('-numChannels',         3,    'Number of channels of the input image')
    cmd:option('-numFeatures',         512,    'Number of features for extraction')
    ------------- Training options --------------------
    cmd:option('-nEpochs',             30,    'Number of total epochs to run')
    cmd:option('-epochSize',          3194,    'Number of batches per epoch(total number / stride / batchSize)') --2514
    cmd:option('-epochTestSize',       26,    'Number of batches per epoch(total number / stride / batchSize)')
    cmd:option('-epochNumber',          1,    'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',          128,    'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',         0.9, 'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'resnet', 'Load Resnet model')
    cmd:option('-mode',     'train', 'select -train- for train, -test- for confusion matrix extraction, -features- for feature extraction')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()
local opt = cmd:parse(arg or {})
    opt.save = paths.concat(opt.cache, '' .. os.date():gsub(' ',''))
    return opt
end

return M
