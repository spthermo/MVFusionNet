--------------------------
------Spiros Thermos------
---Visual Computing Lab---
--------------------------
---all rights reserved----

local batchNumber
local top1_epoch, loss_epoch
confMatrix = torch.Tensor(55,55):fill(0)

-- GPU inputs (preallocate)
local inputView_cuda = torch.CudaTensor()
local labels_cuda = torch.CudaTensor()

--Convert input to 3-channel image
local function processImg(input)
	local rgb = torch.Tensor(3, input:size(2), input:size(3)):fill(0)
	for c=1,3 do -- channels
		rgb[{{c},{},{}}] = input
	end
	return rgb
end

function test()
	batchNumber = 0
	
	classes=paths.dir(opt.inputDataPath .. '/' .. opt.dirName)
	table.sort(classes,function(a,b)return a<b end)
	table.remove(classes,1) table.remove(classes,1)

	model:evaluate()
	input = torch.CudaTensor()
	models={}
	batchCounter = 0
	allSamples = 0
	local img_batch = torch.FloatTensor(opt.batchSize, opt.numChannels, opt.imageSize, opt.imageSize)
	local labels_batch = torch.Tensor(opt.batchSize):fill(0)
	for cls_cnt=1,#classes do
		numOfSamples = 0
		models[cls_cnt]={}
		models[cls_cnt] = paths.dir(opt.inputDataPath .. '/' .. opt.dirName .. '/' .. classes[cls_cnt])
		table.sort(models[cls_cnt] ,function(a,b)return a<b end)
		table.remove(models[cls_cnt],1) table.remove(models[cls_cnt],1)
		frms={}
		for models_cnt=1,#models[cls_cnt] do
			frms[models_cnt]={}
			frms[models_cnt] = paths.dir(opt.inputDataPath .. '/' .. opt.dirName .. '/' ..classes[cls_cnt]..'/'..models[cls_cnt][models_cnt])
			table.sort(frms[models_cnt] ,function(a,b)return a<b end)
			table.remove(frms[models_cnt],1) table.remove(frms[models_cnt],1)
			local features=torch.FloatTensor(18,opt.numFeatures):fill(0) --#frms[models_cnt]
			gather_cnt=0
			total_cnt=0
			for frame_num=1,#frms[models_cnt] do
				if batchCounter < opt.batchSize then
					batchCounter  = batchCounter + 1
					numOfSamples = numOfSamples + 1
					img = image.load(opt.inputDataPath .. '/' .. opt.dirName .. '/' ..classes[cls_cnt] .. '/' .. models[cls_cnt][models_cnt] .. '/' .. frms[models_cnt][frame_num])--read image
					img = processImg(img)
					img_batch[batchCounter] = img
					labels_batch[batchCounter] = cls_cnt
				else
					testBatch(img_batch, labels_batch)
				batchCounter = 0
				img_batch:fill(0)
				labels_batch:fill(0)
				end
			end
		end
		if batchCounter > 0 then
			testBatch(img_batch, labels_batch)
			batchCounter = 0
			img_batch:fill(0)
			labels_batch:fill(0)
		end
		allSamples = allSamples + numOfSamples
		print('Num of samples in class: ' .. numOfSamples)
	end
	cutorch.synchronize()
	local diag = 0
	local sum = 0
	for i=1, confMatrix:size(1) do
		for j=1,confMatrix:size(2) do
			if i == j then diag = diag + confMatrix[i][j] end
			sum = sum + confMatrix[i][j]
		end
	end
	print('Total Samples Passed: ' .. allSamples)
	local overall = (diag / sum)*100
	print('Overall Accuracy: ' .. overall)

	-- save confusion matrix
	local extractResult = assert(io.open("./result.csv", "w"))
	local splitter = ","
	for i=1, confMatrix:size(1) do
		for j=1,confMatrix:size(2) do
			extractResult:write(confMatrix[i][j])
			if j == confMatrix:size(2) then
				extractResult:write("\n")
			else
				extractResult:write(splitter)
			end
		end
	end
	io.close(extractResult)
	collectgarbage()
	return top1_epoch
end

function testBatch(inputsCPUObject, labelsCPU)
	cutorch.synchronize()
	collectgarbage()
	--transfer over to GPU
	inputView_cuda:resize(inputsCPUObject:size()):copy(inputsCPUObject)
	labels_cuda:resize(labelsCPU:size()):copy(labelsCPU)
	local err, feats1, feats2
	local N = opt.batchSize;
	local top1 = 0

	out = model:forward(inputView_cuda)
	batchNumber = batchNumber + 1
	local _,prediction_sorted = out:float():sort(2, true) -- descending
	for i=1,N do
		if labels_cuda[i] > 0 then
			confMatrix[labels_cuda[i]][prediction_sorted[i][1]] = confMatrix[labels_cuda[i]][prediction_sorted[i][1]] + 1
		end
	end
end
