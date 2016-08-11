require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'mnist'
require 'pl'
require 'paths'

utils = dofile('utils.lua')
xtorch = dofile('xtorch.lua')

cmdopt = lapp[[
    --lr                    (default 0.001)             learning rate
    --nhorse                (default 1)                 nb of threads to load data
    --batchSize             (default 128)               batch size
    --nEpoch                (default 200)               nb of epochs
    --backend               (default CPU)               use CPU/GPU
    --nGPU                  (default 2)                 nb of GPUs to use
    --resume                                            resume from checkpoint
    --verbose                                           show debug information
]]

------------------------------------------------
-- 1. prepare data
--
geometry = {32,32}
trainData = mnist.loadTrainSet(60000, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(10000, geometry)
testData:normalizeGlobal(mean, std)

X_train = trainData.data
Y_train = trainData.labels
X_test = testData.data
Y_test = testData.labels

dofile('plaindataset.lua')
ds = PlainDataset({
    X_train = X_train,
    Y_train = Y_train,
    X_test = X_test,
    Y_test = Y_test
})

------------------------------------------------
-- 2. define net
--
net = nn.Sequential()
net:add(nn.Reshape(1024))
net:add(nn.Linear(1024, 512))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.2))
net:add(nn.Linear(512, 512))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.2))
net:add(nn.Linear(512, 10))

------------------------------------------------
-- 3. init optimization params
--
optimState = {
    learningRate = 0.001,
    learningRateDecay = 1e-7,
    weightDecay = 1e-4,
    momentum = 0.9,
    nesterov = true,
    dampening = 0.0
}

opt = {
    ----------- net options --------------------
    net = net,
    ----------- data options -------------------
    dataset = ds,
    nClass = 10,
    ----------- optimization options -----------
    optimizer = optim.sgd,
    criterion = nn.CrossEntropyCriterion,
    optimState = optimState
}

opt = utils.merge(opt, cmdopt)

------------------------------------------------
-- 4. and fit!
--
xtorch.fit(opt)
