# xtorch: Torch extension for easy model training & test

xtorch.fit(...) with:
- model options
    - net: network to be fit
- data options:
    - X_train: training samples
    - Y_train: training targets
    - X_test: test samples
    - Y_test: test targets
- training options:
    - batchSize: batch size
    - nEpoch: nb of epochs to train
    - nClass: nb of classes
- optimization options:
    - optimizer: optimization algorithm
    - optimState: optimization params
    - criterion: criterion defined
- verbose: show debug info

TODO: replace X,Y with data provider/loader.
