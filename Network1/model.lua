
model =  torch.load('models/resnet/resnet-18.t7') 
model:remove(model:size())
model:add(nn.Linear(512,55))
model:cuda()

if opt.backend == 'cudnn' then
	require 'cudnn'
	cudnn.convert(model, cudnn)
elseif opt.backend ~= 'nn' then
	error'Unsupported backend'
end
criterion = nn.CrossEntropyCriterion()
criterion:cuda()
collectgarbage()