local xtorch = {}

function xtorch.fit(opt)
    xtorch.init(opt)

    for i = 1, opt.nEpoch do
        xtorch.train(opt)
        xtorch.test(opt)
    end
end

----------------------------------------------------------------
-- init global params
--
function xtorch.init(opt)
    net = utils.MSRinit(opt.net)
    parameters, gradParameters = net:getParameters()
    criterion = nn.CrossEntropyCriterion()
    confusion = optim.ConfusionMatrix(opt.nClass)
end

----------------------------------------------------------------
-- training
--
function xtorch.train(opt)
    net:training()

    -- parse arguments
    local X = opt.X_train
    local Y = opt.Y_train
    local nEpoch = opt.nEpoch
    local batchSize = opt.batchSize
    local optimState = opt.optimState
    local c = require 'trepl.colorize'

    -- epoch tracker
    epoch = (epoch or 0) + 1
    print(string.format(c.Cyan 'Epoch %d/%d', epoch, nEpoch))

    -- do one epoch
    local indices = torch.randperm(X:size(1)):long():split(batchSize)
    indices[#indices] = nil

    local loss = 0
    for k, v in pairs(indices) do
        local inputs = X:index(1,v)
        local targets = Y:index(1,v)

        feval = function(x)
            if x~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()

            local outputs = net:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            net:backward(inputs, df_do)
            loss = loss + f

            -- display progress, averaged loss & accuracy
            confusion:batchAdd(outputs, targets)
            confusion:updateValids()
            utils.progress(k, #indices, loss/k, confusion.totalValid)
            
            return f, gradParameters
        end

        optim.sgd(feval, parameters, optimState)
    end

    if opt.verbose then print(confusion) end
    confusion:zero() -- reset confusion for test
end

----------------------------------------------------------------
-- test
--
function xtorch.test(opt)
    net:evaluate()

    -- parse arguments
    local X = opt.X_test
    local Y = opt.Y_test
    local batchSize = opt.batchSize

    -- test over the given dataset
    local indices = torch.randperm(X:size(1)):long():split(batchSize)
    indices[#indices] = nil

    local loss = 0
    for k, v in pairs(indices) do
        local inputs = X:index(1,v)
        local targets = Y:index(1,v)
        local outputs = net:forward(inputs)
        local f = criterion:forward(outputs, targets)
        loss = loss + f

        -- display progress
        confusion:batchAdd(outputs, targets)
        confusion:updateValids()
        utils.progress(k, #indices, loss/k, confusion.totalValid)
    end

    if opt.verbose then print(confusion) end
    confusion:zero()
end

return xtorch
