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

__golden__ = {
    "kernel": {
        "depth_to_space": "depth_to_space_golden"
    }
}


def depth_to_space_golden(x, *, block_size, mode="DCR", data_format="NHWC", **kwargs):
    """
    Golden function for DepthToSpace.

    Rearranges elements from the depth dimension into spatial blocks.

    Args:
        x: Input tensor
        block_size: Size of the spatial block (int)
        mode: "DCR" (depth-column-row) or "CRD" (column-row-depth), default "DCR"
        data_format: "NHWC" or "NCHW", default "NHWC"

    Returns:
        Output tensor with rearranged data
    """
    bs = block_size

    if data_format == "NHWC" or data_format == "nhwc":
        N, H, W, C = x.shape
        C_out = C // (bs * bs)

        if mode == "DCR" or mode == "dcr":
            out = x.reshape(N, H, W, bs, bs, C_out)
            out = out.transpose(0, 1, 3, 2, 4, 5)
            out = out.reshape(N, H * bs, W * bs, C_out)
        else:  # CRD mode
            out = x.reshape(N, H, W, C_out, bs, bs)
            out = out.transpose(0, 1, 4, 2, 5, 3)
            out = out.reshape(N, H * bs, W * bs, C_out)

    else:  # NCHW
        N, C, H, W = x.shape
        C_out = C // (bs * bs)

        if mode == "DCR" or mode == "dcr":
            out = x.reshape(N, bs, bs, C_out, H, W)
            out = out.transpose(0, 3, 4, 1, 5, 2)
            out = out.reshape(N, C_out, H * bs, W * bs)
        else:  # CRD mode
            out = x.reshape(N, C_out, bs, bs, H, W)
            out = out.transpose(0, 1, 4, 2, 5, 3)
            out = out.reshape(N, C_out, H * bs, W * bs)

    return out