--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'torch'

local tnt = require 'torchnet-vision.env'
local doc = require 'argcheck.doc'

doc[[]]

require 'torchnet-vision.image'
require 'torchnet-vision.image.transformimage'

require 'torchnet-vision.test.test'

return tnt.vision
