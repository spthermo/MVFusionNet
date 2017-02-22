--------------------------
------Spiros Thermos------
---Visual Computing Lab---
--------------------------
---all rights reserved----

--Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
trainpLogger = optim.Logger(paths.concat(opt.save, 'trainp.log'))
lossLogger = optim.Logger(paths.concat(opt.save, 'loss.log'))
local batchNumber
local top1_epoch, loss_epoch

--Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimStateModel = {
learningRate = opt.LR,
learningRateDecay = 0.0,
momentum = opt.momentum,
dampening = 0.0,
weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
print('Loading optimState from file: ' .. opt.optimState)
optimState = torch.load(opt.optimState)
end

local function paramsForEpoch(epoch)
	if opt.LR ~= 0.0 then -- if manually specified
		return { }
	end
	local regimes = 
	{
		-- start, end,    LR,   WD,
		{  1,     20,   5e-4,   5e-4, },
		{ 21,     30,   1e-4,   0}
	}
	for _, row in ipairs(regimes) do
		if epoch >= row[1] and epoch <= row[2] then
			return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
		end
	end
end

--Convert input to 3-channel image
local function processImg(input)
	local rgb = torch.Tensor(3, input:size(2), input:size(3)):fill(0)
	for c=1,3 do -- channels
		rgb[{{c},{},{}}] = input
	end
	return rgb
end

--GPU inputs (preallocate)
local inputView_cuda = torch.CudaTensor()
local labels_cuda = torch.CudaTensor()
local timer = torch.Timer()
local dataTimer = torch.Timer()
local parametersModel, gradParametersModel = model:getParameters()

function train()
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch)

	classes = paths.dir(opt.inputDataPath .. '/' .. opt.dirName)
	table.sort(classes,function(a,b)return a<b end)
	table.remove(classes,1) table.remove(classes,1)

	models={}
	mat_inds={}
	mat_models_count_per_class=torch.FloatTensor(#classes):fill(0)

	--Load models
	for i=1,#classes do
		models[i]={}
		models[i]=paths.dir(opt.inputDataPath .. '/' .. opt.dirName .. '/' .. classes[i])
		table.sort(models[i],function(a,b)return a<b end)
		table.remove(models[i],1) table.remove(models[i],1)
		for j=1,#models[i] do
			tempPath = models[i][j]
			models[i][j]={}
			models[i][j]=paths.dir(opt.inputDataPath .. '/' .. opt.dirName .. '/' .. classes[i] .. '/'.. tostring(tempPath))
			table.sort(models[i][j],function(a,b)return a<b end)
			table.remove(models[i][j],1) table.remove(models[i][j],1)
		end
		mat_models_count_per_class[i]=#models[i]
		mat_inds[i]=torch.randperm(mat_models_count_per_class[i])
	end

	local total_files = 0
	local total_models = 0
	total_files=torch.sum(mat_models_count_per_class) * 18
	total_models=torch.sum(mat_models_count_per_class)
	print('Total Views: '..total_files)
	pickInputModel={}

	files_cnt=0
	for i=1,#classes do
		for j=1,mat_models_count_per_class[i] do
			table.insert(pickInputModel,i)
		end
	end
	mat_views_per_class = torch.FloatTensor(#classes):fill(0)

	--Balance ShapNetCore training set
	avg_d = 18*total_models/#classes
	D_base = torch.FloatTensor(#classes):fill(0)
	D = torch.FloatTensor(#classes):fill(0)
	D_balanced = torch.FloatTensor(#classes):fill(0)
	for i=1,#classes do
		mat_views_per_class[i] = mat_models_count_per_class[i] * 18
		D_base[i] = mat_views_per_class[i]/avg_d
	end
	D = torch.pow(D_base,0.5) -- threshold=0.5 for category balancing, set it to 0.3 for subcategory balancing
	for i=1,#classes do
		if mat_views_per_class[i] > avg_d then
			D_balanced[i] = avg_d*D[i]
		else
			D_balanced[i] = mat_views_per_class[i]
		end
	end
	D_balanced:ceil()
	total_balanced_views=torch.sum(D_balanced)
	print('Total Views Balanced: ' ..total_balanced_views)

	local params, newRegime = paramsForEpoch(epoch)
	if newRegime then
		optimStateModel2 = 
		{
			learningRate = params.learningRate,
			learningRateDecay = 0.0,
			momentum = opt.momentum,
			dampening = 0.0,
			weightDecay = params.weightDecay
		}
	end
	cutorch.synchronize()

	model:training()
	local tm = torch.Timer()
	top1_epoch = 0
	loss_epoch = 0
	local rand_cnts=torch.randperm(total_models)
	batchNumber = 0
	files_cnt=0
	models_cnt=0
	local inputView = torch.Tensor(opt.batchSize, 3, opt.imageSize, opt.imageSize):fill(0)
	local labels = torch.Tensor(opt.batchSize):fill(0)

	for i=1,opt.epochSize do
		for j=1,opt.batchSize do
			if files_cnt>=total_balanced_views then break end
			files_cnt=files_cnt+1
			class_pick = torch.randperm(total_models)
			classIter = 1
			repeat
				mat_pick_class = pickInputModel[class_pick[classIter]]
				mat_total_count = #models[mat_pick_class]
				classIter = classIter + 1
			until(mat_total_count>0)
			d_diff = 18*mat_models_count_per_class[mat_pick_class] - mat_views_per_class[mat_pick_class]
			classIter = 2
			while D_balanced[mat_pick_class] < d_diff do
				repeat
					mat_pick_class = pickInputModel[class_pick[classIter]]
					mat_total_count = #models[mat_pick_class]
					classIter = classIter + 1
				until(mat_total_count>0)
				d_diff = 18*mat_models_count_per_class[mat_pick_class] - mat_views_per_class[mat_pick_class]
			end
			vid_pick = torch.randperm(mat_total_count)
			frame_total_num = #models[mat_pick_class][vid_pick[1]]
			frame_pick = torch.randperm(frame_total_num)
			
			--Keep first part of sting (it is actually the model index)
			local t={};counter=1; sep="_"
			local inputstr = tostring(models[mat_pick_class][vid_pick[1]][frame_pick[1]])
			for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
				t[counter]=str
				counter=counter+1
			end
			
			--Load random image (from 18 possible)
			viewDir = opt.inputDataPath..'/' .. opt.dirName .. '/' .. classes[mat_pick_class] .. '/' .. t[1] .. '/' .. models[mat_pick_class][vid_pick[1]][frame_pick[1]]
			viewFeatures = image.load(viewDir)
			viewFeatures = processImg(viewFeatures)
			
			--Remove the loaded image index
			table.remove(models[mat_pick_class][vid_pick[1]],1)
			mat_views_per_class[mat_pick_class] = mat_views_per_class[mat_pick_class] - 1
			if #models[mat_pick_class][vid_pick[1]] < 1 then 
				table.remove(models[mat_pick_class], vid_pick[1])
				table.remove(pickInputModel, class_pick[1])
				total_models = total_models-1
			end
			inputView[j]:copy(viewFeatures)
			labels[j] = mat_pick_class
		end
		trainBatch(inputView, labels)
	end

	cutorch.synchronize()

	top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize ) -- * opt.depthSize
	loss_epoch = loss_epoch / opt.epochSize

	trainLogger:add
	{
		['% top1 accuracy (train set)'] = top1_epoch,
		['avg loss (train set)'] = loss_epoch
	}
	lossLogger:add{ ['Loss'] = loss_epoch*100 }
	lossLogger:style{['Loss'] = '-'}
	lossLogger:plot()
	print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
			.. 'average loss (per batch): %.2f \t '
			.. 'accuracy(%%):\t top-1 %.2f\t',
	epoch, tm:time().real, loss_epoch, top1_epoch))
	print('\n')
	collectgarbage()
	-- clear the intermediate states in the model before saving to disk
	model:clearState()

	if epoch == opt.nEpochs then 
		saveDataParallel(paths.concat(opt.save, 'model_after_epoch_' .. epoch .. '.t7'), model)
	end
	return top1_epoch
end

function trainBatch(inputsCPUObject, labelsCPU)
	cutorch.synchronize()
	collectgarbage()
	local dataLoadingTime = dataTimer:time().real
	local err
	local N = opt.batchSize;
	local top1 = 0
	timer:reset()
	inputView_cuda:resize(inputsCPUObject:size()):copy(inputsCPUObject)
	labels_cuda:resize(labelsCPU:size()):copy(labelsCPU)

	model:zeroGradParameters()
	out = model:forward(inputView_cuda) --out1
	err = criterion:forward(out, labels_cuda)
	local gradOut = criterion:backward(out, labels_cuda)
	model:backward(inputView_cuda, gradOut:cuda())
	fevalModel = function()
		return err, gradParametersModel
	end
	optim.sgd(fevalModel, parametersModel, optimStateModel)

	if model.needsSync then
		model:syncParameters()
	end
	cutorch.synchronize()
	batchNumber = batchNumber + 1
	loss_epoch = loss_epoch + err
	do
		local _,prediction_sorted = out:float():sort(2, true) -- descending
		for i=1,N do
			if prediction_sorted[i][1] == labelsCPU[i] then
				top1_epoch = top1_epoch + 1
				top1 = top1 + 1
			end
		end
		top1 = top1 * 100 / N
	end
	print(('Epoch: [%d][%d/%d] Time %.3f Err %.4f Top1-%%: %.2f(%.2f) LR %.0e DataLoadingTime %.3f'):format(epoch, batchNumber, opt.epochSize, timer:time().real, err, top1, top1_epoch * 100 / (batchNumber * opt.batchSize ), optimStateModel2.learningRate, dataLoadingTime))
	dataTimer:reset()
end
