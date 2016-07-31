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
    learningRate = 0.01,
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
    X_train = X_train,
    Y_train = Y_train,
    X_test = X_test,
    Y_test = Y_test,
    ----------- training options ---------------
    batchSize = 128,
    nEpoch = 5,
    nClass = 10,
    ----------- optimization options -----------
    optimizer = 'SGD',
    criterion = 'CrossEntropyCriterion',
    optimState = optimState,
    ----------- general options ----------------
    verbose = false
}

------------------------------------------------
-- 4. and fit!
--
xtorch.fit(opt)
