local Image = require('image')




function split(s, delim)
  if type(delim) ~= "string" or string.len(delim) <= 0 then
    return
  end

  local start = 1
  local t = {}
  while true do
  local pos = string.find (s, delim, start, true) -- plain find
    if not pos then
     break
    end

    table.insert (t, string.sub (s, start, pos - 1))
    start = pos + string.len (delim)
  end
  table.insert (t, string.sub (s, start))

  return t
end




function str2label(strs, maxLength)
    --[[ Convert a list of strings to integer label tensor (zero-padded).

    ARGS:
      - `strs`     : table, list of strings
      - `maxLength`: int, the second dimension of output label tensor

    RETURN:
      - `labels`   : tensor of shape [#(strs) x maxLength]
    ]]
    assert(type(strs) == 'table')

    function ascii2label(ascii)
        local label
        if ascii >= 48 and ascii <= 57 then -- '0'-'9' are mapped to 1-10
            label = ascii - 47
        elseif ascii >= 65 and ascii <= 90 then -- 'A'-'Z' are mapped to 11-36
            label = ascii - 64 
        elseif ascii >= 97 and ascii <= 122 then -- 'a'-'z' are mapped to 11-36
            label = ascii - 96 
        end
        return label
    end


    local nStrings = #strs
    local labels = torch.IntTensor(nStrings, maxLength):fill(0)
    for i, str in ipairs(strs) do
	tmp=split(str," ")
	for j = 1,table.getn(tmp)-1 do
		labels[i][j]=tmp[j]+1
	end
	--[[
        for j = 1, string.len(str) do
            local ascii = string.byte(str, j)
            labels[i][j] = ascii2label(ascii)
        end ]]
    end
    return labels
end


function label2str(labels, raw)
    --[[ Convert a label tensor to a list of strings.

    ARGS:
      - `labels`: int tensor, labels
      - `raw`   : boolean, if true, convert zeros to '-'

    RETURN:
      - `strs`  : table, list of strings
    ]]
    assert(labels:dim() == 2)
    raw = raw or false

    function label2ascii(label)
        local ascii
        if label >= 1 and label <= 10 then
            ascii = label - 1 + 48
        elseif label >= 11 and label <= 36 then
            ascii = label - 11 + 97
        elseif label == 0 then -- used when displaying raw predictions
            ascii = string.byte('-')
        end
        return ascii
    end

    local strs = {}
    local nStrings, maxLength = labels:size(1), labels:size(2)
    for i = 1, nStrings do
        local str = {}
        local ss = " "
        local labels_i = labels[i]
        for j = 1, maxLength do
            if raw then
                str[j] = tostring(labels_i[j]) --label2ascii(labels_i[j])
            else
                if labels_i[j] == 0 then
                    break
                else
                    str[j] = tostring(labels_i[j]) -- label2ascii(labels_i[j])
                end
            end
	    ss = ss..str[j].." "
        end
        str = unpack(str)
        strs[i] = ss
    end
    return strs
end


function setupLogger(fpath)
    local fileMode = 'w'
    if paths.filep(fpath) then
        local input = nil
        while not input do
            print('Logging file exits, overwrite(o)? append(a)? abort(q)?')
            -- input = io.read()
            input = 'o'
            if input == 'o' then
                fileMode = 'w'
            elseif input == 'a' then
                fileMode = 'a'
            elseif input == 'q' then
                os.exit()
            else
                fileMode = nil
            end
        end
    end
    gLoggerFile = io.open(fpath, fileMode)
end


function tensorInfo(x, name)
    local name = name or ''
    local sizeStr = ''
    for i = 1, #x:size() do
        sizeStr = sizeStr .. string.format('%d', x:size(i))
        if i < #x:size() then
            sizeStr = sizeStr .. 'x'
        end
    end
    infoStr = string.format('[%15s] size: %12s, min: %+.2e, max: %+.2e', name, sizeStr, x:min(), x:max())
    return infoStr
end


function shutdownLogger()
    if gLoggerFile then
        gLoggerFile:close()
    end
end


function logging(message, mute)
    mute = mute or false
    local timeStamp = os.date('%x %X')
    local msgFormatted = string.format('[%s]  %s', timeStamp, message)
    if not mute then
        print(msgFormatted)
    end
    if gLoggerFile then
        gLoggerFile:write(msgFormatted .. '\n')
        gLoggerFile:flush()
    end
end


function modelSize(model)
    local params = model:parameters()
    local count = 0
    local countForEach = {}
    for i = 1, #params do
        local nParam = params[i]:numel()
        count = count + nParam
        countForEach[i] = nParam
    end
    return count, torch.LongTensor(countForEach)
end


function cloneList(tensors, fillZero)
    --[[ Clone a list of tensors, adapted from https://github.com/karpathy/char-rnn
    ARGS:
      - `tensors`  : table, list of tensors
      - `fillZero` : boolean, if true tensors are filled with zeros
    RETURNS:
      - `output`   : table, cloned list of tensors
    ]]
    local output = {}
    for k, v in pairs(tensors) do
        output[k] = v:clone()
        if fillZero then output[k]:zero() end
    end
    return output
end


function cloneManyTimes(net, T)
    --[[ Clone a network module T times, adapted from https://github.com/karpathy/char-rnn
    ARGS:
      - `net`    : network module to be cloned
      - `T`      : integer, number of clones
    RETURNS:
      - `clones` : table, list of clones
    ]]
    local clones = {}
    local params, gradParams = net:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        if params then
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end


function diagnoseGradients(params, gradParams)
    --[[ Diagnose gradients by checking the value range and the ratio of the norms
    ARGS:
      - `params`     : first arg returned by net:parameters()
      - `gradParams` : second arg returned by net:parameters()
    ]]
    for i = 1, #params do
        local pMin = params[i]:min()
        local pMax = params[i]:max()
        local gpMin = gradParams[i]:min()
        local gpMax = gradParams[i]:max()
        local normRatio = gradParams[i]:norm() / params[i]:norm()
        logging(string.format('%02d - params [%+.2e, %+.2e] gradParams [%+.2e, %+.2e], norm gp/p %+.2e',
            i, pMin, pMax, gpMin, gpMax, normRatio), true)
    end
end


function modelState(model)
    --[[ Get model state, including model parameters (weights and biases) and
         running mean/var in batch normalization layers
    ARGS:
      - `model` : network model
    RETURN:
      - `state` : table, model states
    ]]
    local parameters = model:parameters()
    local bnVars = {}
    local bnLayers = model:findModules('nn.BatchNormalization')
    for i = 1, #bnLayers do
        bnVars[#bnVars+1] = bnLayers[i].running_mean
        bnVars[#bnVars+1] = bnLayers[i].running_var
    end
    local bnLayers = model:findModules('nn.SpatialBatchNormalization')
    for i = 1, #bnLayers do
        bnVars[#bnVars+1] = bnLayers[i].running_mean
        bnVars[#bnVars+1] = bnLayers[i].running_var
    end
    local state = {parameters = parameters, bnVars = bnVars}
    return state
end


function loadModelState(model, stateToLoad)
    local state = modelState(model)
    assert(#state.parameters == #stateToLoad.parameters)
    assert(#state.bnVars == #stateToLoad.bnVars)
    for i = 1, #state.parameters do
        state.parameters[i]:copy(stateToLoad.parameters[i])
    end
    for i = 1, #state.bnVars do
        state.bnVars[i]:copy(stateToLoad.bnVars[i])
    end
end


function add_border(img)
	local WIDTH = 480
    local HEIGHT = 32
    local img_shape = img:size()
	local img_width = img_shape[3]
	local img_height = img_shape[2]
    local img_channels = img_shape[1]

	if img_width < WIDTH then
        local new_img = torch.ByteTensor(3, HEIGHT, WIDTH):fill(255)
		local left_border = (WIDTH - img_width) / 2
        new_img[{ {1, 3}, {1,HEIGHT}, {left_border, left_border + img_width - 1}}]= img
        return new_img
    elseif img_width > WIDTH then  
        print("img_width is bigger than 480, img_width = " .. img_width)
        img = image.scale(img, WIDTH, HEIGHT)
        return img
    else
        return img
	end
end

function loadAndResizeImage(imagePath)
    local ok, img = pcall(Image.load, imagePath, 3, 'byte')
    if not ok then
        print('invalid image: ',imagePath)
        return nil
    end

    local resized_proportionally = 1 
    if resized_proportionally == 0 then
        img = image.scale(img, 480, 32)
    else
        img_size = img:size()
        resized_width = 1.0 * img_size[3] / img_size[2] * 32
        img = image.scale(img, resized_width, 32)
        img = add_border(img)
    end

    return img
end


function recognizeImageLexiconFree(model, image)
    --[[ Lexicon-free text recognition.
    ARGS:
      - `model`   : CRNN model
      - `image`   : single-channel image, byte tensor
    RETURN:
      - `str`     : recognized string
      - `rawStr`  : raw recognized string
    ]]
    --assert(image:dim() == 2 and image:type() == 'torch.ByteTensor',
    --    'Input image should be single-channel byte tensor')
    image = image:view(1, 3, image:size(2), image:size(3))
    local output = model:forward(image)
    local pred, predRaw = naiveDecoding(output)
    local str = label2str(pred)[1]
    local rawStr = label2str(predRaw, true)[1]
    return str, rawStr
end


function recognizeImageWithLexicion(model, image, lexicon)
    --[[ Text recognition with a lexicon.
    ARGS:
      - `imagePath` : string, image path
      - `lexicon`   : list of string, lexicon words
    RETURN:
      - `str`       : recognized string
    ]]
    assert(image:dim() == 2 and image:type() == 'torch.ByteTensor',
        'Input image should be single-channel byte tensor')
    image = image:view(1, 1, image:size(1), image:size(2))
    local output = model:forward(image)
    local str = decodingWithLexicon(output, lexicon)
    return str
end
