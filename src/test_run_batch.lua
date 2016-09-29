--Author: hongji
--Date: 2016-06-04

require('nn')

require('cutorch')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')


require('lfs')
function getfilelist(path, file_table)
    print('get file list ...')
    local cc = 0
    for file in lfs.dir(path) do
        if file ~= '.' and file ~= '..' then
            cc = cc + 1
            --print(cc,file)
            local uri = path..'/'..file
            --print(cc,uri)
            table.insert(file_table, uri)
        end
    end
    table.sort(file_table)
    return 
end


function make_dict(duri, id2char_tb)
    print('make dict ...')
    fp = io.open(duri, 'r')
    assert(fp)
    cc = 0
    for line in fp:lines() do
        --print(line)
        cc = cc + 1
        id2char_tb[cc] = line
    end
    fp:close()
    print('dict size:',#id2char_tb)
end


local dev_id = tonumber(arg[1])
local path = arg[2]
local resuri = arg[3]
local wordlisturi = 'word_3567.txt'
local id2char_tb = {}
make_dict(wordlisturi, id2char_tb)


local img_tb ={}
getfilelist(path, img_tb)  -- get images in path
print('img num:',#img_tb)

cutorch.setDevice(dev_id)
torch.setnumthreads(2)
torch.setdefaulttensortype('torch.FloatTensor')

--print('Loading model...')
-- Test 1
--local modelDir = '/data/zhangyg/work_torch/model/crnn/3567_5m_4color_7font_shadow_480'
--paths.dofile(paths.concat(modelDir, 'config.lua'))
--local modelLoadPath = paths.concat(modelDir, 'snapshot_168000.t7')
local modelDir = '../model/'
paths.dofile(paths.concat(modelDir, 'config.lua'))
--local modelLoadPath = paths.concat(modelDir, 'snapshot_174000.t7')i
local modelName = arg[4]
local modelLoadPath = paths.concat(modelDir,modelName) 

gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
--print(modelLoadPath)
local snapshot = torch.load(modelLoadPath)
--print ('snapshot end')
loadModelState(model, snapshot)
model:evaluate()
--print(string.format('Model loaded from %s', modelLoadPath))


print('do predict')
t0 = os.clock()
fpw = io.open(resuri, 'w')
assert(fpw)
for i=1,#img_tb do
    local imagePath = img_tb[i]

    local img = loadAndResizeImage(imagePath)
    if img == nil then
        print('invalid: ',imagePath) 
    else
        local text, raw = recognizeImageLexiconFree(model, img)
        --print(string.format('Recognized text: %s (raw: %s)', text, raw))
        --print(text)
        local resstr = ''
        ch_tb = split(text, ' ')
         
        for k = 1,#ch_tb do
            cid = tonumber(ch_tb[k])
            if cid ~= nil then
                --print(cid)
                local char = id2char_tb[cid-1] --first cid is space
                --print(char)
                resstr = resstr..char
            end
        end
        
        local ind = string.find(imagePath, '/[^/]*$')
        local imgname = string.sub(imagePath, ind+1)
        print(i,imgname, resstr)
        resline = string.format("%s: %s\n", imgname,resstr) 
        fpw:write(resline)
        if i > 2000 then
            break 
        end
    end
end
fpw:close()
print(string.format('time cost: %.2fs\n', os.clock()-t0))
