#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy as np
import torch

__golden__ = {
   	"kernel": {
   	    "floor_mod": "floor_mod_golden"
   	}
}

def numpy_to_torch_tensor(x_np):
    """
    Convert numpy array to torch tensor, handling bfloat16 specifically.
    """
    dtype_name = x_np.dtype.name
    if dtype_name == "bfloat16":
        return torch.from_numpy(x_np.astype(np.float32)).to(torch.bfloat16)
    else:
        return torch.from_numpy(x_np)

def floor_mod_golden(x1, x2,
                     **kwargs):
    '''
    Kernel golden for floor_mod.
    All the parameters follow @floor_mod_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    type_int = [torch.int64, torch.int32]
    type_float = [torch.float, torch.float16, torch.bfloat16]

    # 除零保护
    res_shape = np.broadcast_shapes(x1.shape, x2.shape)
    X2_broadcast = np.broadcast_to(x2, res_shape)
    zero_X2_broadcast_idx = np.where(X2_broadcast == 0)

    zero_idx = np.where(x2 == 0)
    if zero_idx:
        x2[zero_idx] = 1
    
    x1_dtype = x1.dtype
    x1 = numpy_to_torch_tensor(x1)
    x2 = numpy_to_torch_tensor(x2)
    res = torch.remainder(x1, x2)

    # 除零保护
    if zero_idx:
        x2[zero_idx] = 0
        if res.dtype in type_int:
            res[zero_X2_broadcast_idx] = -1
        if res.dtype in type_float:
            res[zero_X2_broadcast_idx] = torch.nan

    if res.dtype == torch.bfloat16:
        res_np = res.float().numpy()
    else:
        res_np = res.numpy()

    return res_np.astype(x1_dtype, copy=False)
