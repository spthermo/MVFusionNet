--------------------------
------Spiros Thermos------
---Visual Computing Lab---
--------------------------
---all rights reserved----

--Convert input to 3-channel image
local function processImg(input)
    local rgb = torch.Tensor(3, input:size(2), input:size(3)):fill(0)
    for c=1,3 do -- channels
      rgb[{{c},{},{}}] = input
    end
    return rgb
end

if opt.dirName == 'testSet' then
    numOfClasses = 1
else
    classes=paths.dir(opt.inputDataPath .. '/' .. opt.dirName)
    table.sort(classes,function(a,b)return a<b end)
    table.remove(classes,1) table.remove(classes,1)
    numOfClasses = #classes
end

if not paths.dirp(opt.targetDirName) then
    paths.mkdir(opt.targetDirName)
end

if (opt.dirName == 'trainSet') or (opt.dirName == 'valSet') then
    for i=1,numOfClasses do
        if not paths.dirp(opt.targetDirName .. '/' .. classes[i]) then
            paths.mkdir(opt.targetDirName .. '/' .. classes[i])
        end
    end
end


model:remove(model:size())
model:cuda()

model:evaluate()

function features()
    input = torch.CudaTensor()
    models={}
    for cls_cnt=1, numOfClasses do
        models[cls_cnt]={}
        if opt.dirName == 'testSet' then
            models[cls_cnt] = paths.dir(opt.inputDataPath .. '/' .. opt.dirName) --
        else
            models[cls_cnt] = paths.dir(opt.inputDataPath .. '/' .. opt.dirName .. '/' .. classes[cls_cnt])
        end
        table.sort(models[cls_cnt] ,function(a,b)return a<b end)
        table.remove(models[cls_cnt],1) table.remove(models[cls_cnt],1)
        frms={}
        for models_cnt=1,#models[cls_cnt] do
            frms[models_cnt]={}
            if opt.dirName == 'testSet' then
                frms[models_cnt] = paths.dir(opt.inputDataPath .. '/' .. opt.dirName .. '/' .. models[cls_cnt][models_cnt])
            else
                frms[models_cnt] = paths.dir(opt.inputDataPath .. '/' .. opt.dirName .. '/' ..classes[cls_cnt] .. '/' .. models[cls_cnt][models_cnt])
            end
            table.sort(frms[models_cnt] ,function(a,b)return a<b end)
            table.remove(frms[models_cnt],1) table.remove(frms[models_cnt],1)
            local features=torch.FloatTensor(18,opt.numFeatures):fill(0)
            gather_cnt=0
            total_cnt=0

            local img_batch = torch.FloatTensor(opt.batchSize, opt.numChannels, opt.imageSize, opt.imageSize)
            for frame_num=1,#frms[models_cnt] do
                if opt.dirName == 'testSet' then
                    img = image.load(opt.inputDataPath .. '/' .. opt.dirName .. '/' .. models[cls_cnt][models_cnt] .. '/' .. frms[models_cnt][frame_num]) --
                else
                    img = image.load(opt.inputDataPath .. '/' .. opt.dirName .. '/' .. classes[cls_cnt] .. '/' .. models[cls_cnt][models_cnt] .. '/' .. frms[models_cnt][frame_num]) 
                end
                img = processImg(img)
                gather_cnt=gather_cnt+1
                img_batch[gather_cnt] = img
                if (gather_cnt==opt.batchSize) then
                    local output = model:forward(img_batch:cuda()):squeeze(1)
                    output=output:float()
                    for i_ins=1,gather_cnt do
                        total_cnt=total_cnt+1
                        features[total_cnt]=output[i_ins]
                    end
                    gather_cnt =0
                    img_batch:fill(0)
                end
            end 
            print(' Video: ' .. models[cls_cnt][models_cnt] .. ' Found frames: ' .. total_cnt+gather_cnt .. '/' .. #frms[models_cnt]) --'Class: ' .. classes[cls_cnt] .. 
            if gather_cnt>0 and gather_cnt<opt.batchSize then
                local output = model:forward(img_batch:cuda()):squeeze(1)
                output=output:float()     
                for i_ins=1,gather_cnt do
                    total_cnt=total_cnt+1
                    features[total_cnt]=output[i_ins]
                end
                gather_cnt =0
                img_batch:fill(0)
            end
            if opt.dirName == 'testSet' then
                torch.save(opt.targetDirName .. '/' .. models[cls_cnt][models_cnt] .. '.t7',features)
            else
                torch.save(opt.targetDirName .. '/' .. classes[cls_cnt] .. '/' .. models[cls_cnt][models_cnt] .. '.t7',features)
            end
        end  
    end
end