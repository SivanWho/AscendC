/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file complex_apt.cpp
 * \brief Complex operator kernel entry
 */

#include "kernel_operator.h"
#include "arch35/complex_struct.h"
#include "arch35/complex.h"

using namespace AscendC;

//   Complex_float32 : DTYPE_REAL=float, DTYPE_OUT=complex64
//   Complex_float16 : DTYPE_REAL=half,  DTYPE_OUT=complex32
#ifndef DTYPE_REAL
#define DTYPE_REAL float
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT complex64
#endif

extern "C" __global__ __aicore__ void complex(
    GM_ADDR real, GM_ADDR imag, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ComplexTilingData);
    GET_TILING_DATA(tilingData, tiling);
    ComplexOp::ComplexSimt op;
    op.Init(real, imag, out, tilingData);
    op.Process();
}