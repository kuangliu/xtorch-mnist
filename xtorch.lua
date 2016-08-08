local xtorch = {}
local opt

function xtorch.fit(opt_)
    opt = opt_
    xtorch.init()

    for i = 1, opt.nEpoch do
        xtorch.train()
        xtorch.test()
    end
end

----------------------------------------------------------------
-- init global params
--
function xtorch.init()
    net = utils.MSRinit(opt.net)
    criterion = nn.CrossEntropyCriterion()

    -- use GPU
    if opt.backend == 'GPU' then
        require 'cunn'
        require 'cutorch'
        cudnn = require 'cudnn'

        cudnn.convert(net, cudnn):cuda()
        cudnn.fastest = true
        cudnn.benchmark = true

        -- insert data augment layers
        local net_ = nn.Sequential()
                      :add(nn.BatchFlip():float())
                      :add(nn.RandomCrop(4, 'zero'):float())
                      :add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))

        if opt.nGPU == 1 then
            cutorch.setDevice(1)
        else
            net = utils.makeDataParallelTable(net, opt.nGPU)
        end
        net_:add(net)
        net = net_
        criterion = criterion:cuda()
    end

    print(net)

    parameters, gradParameters = net:getParameters()
    confusion = optim.ConfusionMatrix(opt.nClass)

    -- data loader
    if opt.nhorse == 1 or opt.nhorse == nil then   -- default single thread
        horses = {}
        function horses:addjob(f1, f2) f2(f1()) end
        function horses:synchronize() end
    else                                           -- multi thread
        threads = require 'threads'
        horses = threads.Threads(opt.nhorse,
            function()
                require 'nn'
                require 'torch'
            end,
            function(idx)
                print('init thread '..idx)
                -- dofile('listdataset.lua')
                dofile('plaindataset.lua')
            end
        )
    end
end

----------------------------------------------------------------
-- cuda synchronize for each epoch/batch training/test.
--
function xtorch.cudaSync()
    if opt.backend=='GPU' then cutorch.synchronize() end
end

----------------------------------------------------------------
-- training
--
function xtorch.train()
    -- cuda sync for each epoch
    xtorch.cudaSync()
    net:training()

    -- parse arguments
    local nEpoch = opt.nEpoch
    local batchSize = opt.batchSize
    local optimState = opt.optimState
    local dataset = opt.dataset
    local c = require 'trepl.colorize'

    -- epoch tracker
    epoch = (epoch or 0) + 1
    print(string.format(c.Cyan 'Epoch %d/%d', epoch, nEpoch))

    -- do one epoch
    trainLoss = 0
    local epochSize = math.floor(dataset.ntrain/opt.batchSize)
    local bs = opt.batchSize
    for i = 1,epochSize do
        horses:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, targets = dataset:sample(bs)
                return inputs, targets
            end,
            -- the end callback (runs in the main thread)
            function (inputs, targets)
                -- cuda sync for each batch
                xtorch.cudaSync()
                -- if use GPU, convert to cuda tensor
                targets = opt.backend=='GPU' and targets:cuda() or targets

                feval = function(x)
                    if x~= parameters then
                        parameters:copy(x)
                    end
                    gradParameters:zero()

                    local outputs = net:forward(inputs)
                    local f = criterion:forward(outputs, targets)
                    local df_do = criterion:backward(outputs, targets)
                    net:backward(inputs, df_do)
                    trainLoss = trainLoss + f

                    -- display progress & loss
                    confusion:batchAdd(outputs, targets)
                    confusion:updateValids()
                    utils.progress(i, epochSize, trainLoss/i, confusion.totalValid)
                    return f, gradParameters
                end
                optim.sgd(feval, parameters, optimState)
                xtorch.cudaSync()
            end
        )
    end

    xtorch.cudaSync()
    horses:synchronize() -- wait all horses back
    if opt.verbose then print(confusion) end
    confusion:zero()     -- reset confusion for test
end

----------------------------------------------------------------
-- test
--
function xtorch.test()
    xtorch.cudaSync()
    net:evaluate()

    local dataset = opt.dataset
    local epochSize = math.floor(dataset.ntest/opt.batchSize)
    local bs = opt.batchSize

    testLoss = 0
    for i = 1, epochSize do
        horses:addjob(
            function()
                local inputs, targets = dataset:get(bs*(i-1)+1, bs*i)
                return inputs, targets
            end,
            function(inputs, targets)
                xtorch.cudaSync()
                targets = opt.backend=='GPU' and targets:cuda() or targets

                local outputs = net:forward(inputs)
                local f = criterion:forward(outputs, targets)
                testLoss = testLoss + f

                -- display progress
                confusion:batchAdd(outputs, targets)
                confusion:updateValids()
                utils.progress(i, epochSize, testLoss/i, confusion.totalValid)
                xtorch.cudaSync()
            end
        )
    end

    xtorch.cudaSync()
    horses:synchronize()
    if opt.verbose then print(confusion) end
    confusion:zero()
    print('\n')
end

return xtorch
