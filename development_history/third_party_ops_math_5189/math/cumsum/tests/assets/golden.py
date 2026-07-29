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
        "cumsum": "cumsum_golden"
    }
}

def cumsum_golden(x, axis, *, exclusive, reverse, **kwargs):
    '''
    Golden function for cumsum.
    All the parameters (names and order) follow @cumsum_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    x_dtype = x.dtype
    if x_dtype.name == "bfloat16" or x_dtype.name == "float16":
        x = x.astype("float32")

    p0 = tf.constant(x)
    out = tf.cumsum(x=p0, axis=axis, reverse=reverse, exclusive=exclusive)

    with tf.Session() as sess:
        res = sess.run(out)
    if x_dtype.name == "bfloat16" or x_dtype.name == "float16":
        res = res.astype(x_dtype, copy=False)
    return res
