require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('training')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')
require('nnx')

-- configurations
print(tonumber(arg[1]))
cutorch.setDevice(tonumber(arg[1]))
--cutorch.setDevice(2)
torch.setnumthreads(10)
torch.setdefaulttensortype('torch.FloatTensor')
local modelDir = arg[2]
setupLogger(paths.concat(modelDir, 'log.txt'))
paths.dofile(paths.concat(modelDir, 'config.lua'))
gConfig = getConfig()
gConfig.modelDir = modelDir

-- `createModel` is defined in config.lua, it returns the network model and the criterion (loss function)
local model, criterion = createModel(gConfig)
logging(string.format('Model configuration:\n%s', model))
local modelSize, nParamsEachLayer = modelSize(model)
logging(string.format('Model size: %d\n%s', modelSize, nParamsEachLayer))

-- load model snapshot
local loadPath = arg[3]
if loadPath then
    local snapshot = torch.load(loadPath)
    loadModelState(model, snapshot)
    logging(string.format('Model loaded from %s', loadPath))
end

-- load dataset
logging('Loading datasets...%s', gConfig.trainSetPath)
local trainSet = DatasetLmdb(gConfig.trainSetPath, gConfig.trainBatchSize)
local valSet = DatasetLmdb(gConfig.valSetPath)

-- train and test model
logging('Start training...')
trainModel(model, criterion, trainSet, valSet)

shutdownLogger()
