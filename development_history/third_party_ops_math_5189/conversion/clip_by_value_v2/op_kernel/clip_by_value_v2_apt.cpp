/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file clip_by_value_v2_apt.cpp
 * \brief
 */

#include "arch35/clip_by_value_v2_ass.h"
#include "arch35/clip_by_value_v2_single_dim.h"
#include "arch35/clip_by_value_v2_ub_broadcast.h"
#include "arch35/clip_by_value_v2_f16_nddma_without_loops.h"
#include "arch35/clip_by_value_v2_f16_nddma_with_loops.h"
#include "arch35/clip_by_value_v2_f32_nddma_without_loops.h"
#include "arch35/clip_by_value_v2_f32_nddma_with_loops.h"
#include "arch35/clip_by_value_v2_bf16_nddma_without_loops.h"
#include "arch35/clip_by_value_v2_bf16_nddma_with_loops.h"
#include "arch35/clip_by_value_v2_s32_nddma_without_loops.h"
#include "arch35/clip_by_value_v2_s32_nddma_with_loops.h"
#include "arch35/clip_by_value_v2_s64_nddma_without_loops.h"
#include "arch35/clip_by_value_v2_s64_nddma_with_loops.h"

using namespace ClipByValueV2;

#define CLIP_BY_VALUE_V2_SINGLE_DIM_TILING_KEY 2000100
#define CLIP_BY_VALUE_V2_ASS_TILING_KEY 2001100

#define CLIP_BY_VALUE_V2_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY 100000001000100
#define CLIP_BY_VALUE_V2_F16_NDDMA_WITH_LOOPS_TILING_KEY 100000001001100

#define CLIP_BY_VALUE_V2_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY 200000001000100
#define CLIP_BY_VALUE_V2_F32_NDDMA_WITH_LOOPS_TILING_KEY 200000001001100

#define CLIP_BY_VALUE_V2_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY 300000001000100
#define CLIP_BY_VALUE_V2_BF16_NDDMA_WITH_LOOPS_TILING_KEY 300000001001100

#define CLIP_BY_VALUE_V2_S32_NDDMA_WITHOUT_LOOPS_TILING_KEY 400000001000100
#define CLIP_BY_VALUE_V2_S32_NDDMA_WITH_LOOPS_TILING_KEY 400000001001100

#define CLIP_BY_VALUE_V2_S64_NDDMA_WITHOUT_LOOPS_TILING_KEY 500000001000100
#define CLIP_BY_VALUE_V2_S64_NDDMA_WITH_LOOPS_TILING_KEY 500000001001100

#define CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_1_TILING_KEY 3000001
#define CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_2_TILING_KEY 3000002
#define CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_3_TILING_KEY 3000003
#define CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_4_TILING_KEY 3000004
#define CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_9_TILING_KEY 3000009

static constexpr int64_t CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_1 = 1;
static constexpr int64_t CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_2 = 2;
static constexpr int64_t CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_3 = 3;
static constexpr int64_t CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_4 = 4;
static constexpr int64_t CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_DYNAMIC = -1;

extern "C" __global__ __aicore__ void clip_by_value_v2(
    GM_ADDR x, GM_ADDR clipValueMin, GM_ADDR clipValueMax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe tPipe;
    if (TILING_KEY_IS(CLIP_BY_VALUE_V2_ASS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueSigDimTilingData, tilingDataIn, tiling);
        const ClipByValueSigDimTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2Ass<DTYPE_X> op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_SINGLE_DIM_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueSigDimTilingData, tilingDataIn, tiling);
        const ClipByValueSigDimTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2SigDim<DTYPE_X> op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2F16NddmaWithoutLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2F16NddmaWithLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2F32NddmaWithoutLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F32_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2F32NddmaWithLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2Bf16NddmaWithoutLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_BF16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2Bf16NddmaWithLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_S32_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2S32NddmaWithoutLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_S32_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2S32NddmaWithLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_S64_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2S64NddmaWithoutLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_S64_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingDataIn, tiling);
        const ClipByValueTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2S64NddmaWithLoops op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_1_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueUbBrcTilingData, tilingDataIn, tiling);
        const ClipByValueUbBrcTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2UbBroadcast<DTYPE_X, CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_1> op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_2_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueUbBrcTilingData, tilingDataIn, tiling);
        const ClipByValueUbBrcTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2UbBroadcast<DTYPE_X, CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_2> op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_3_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueUbBrcTilingData, tilingDataIn, tiling);
        const ClipByValueUbBrcTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2UbBroadcast<DTYPE_X, CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_3> op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_4_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueUbBrcTilingData, tilingDataIn, tiling);
        const ClipByValueUbBrcTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2UbBroadcast<DTYPE_X, CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_4> op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_9_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(ClipByValueUbBrcTilingData, tilingDataIn, tiling);
        const ClipByValueUbBrcTilingData* __restrict tilingData = &tilingDataIn;
        ClipByValueV2UbBroadcast<DTYPE_X, CLIP_BY_VALUE_V2_F16_UB_BROADCAT_RANK_DYNAMIC> op;
        op.Init(x, clipValueMin, clipValueMax, y, workspace, tilingData, &tPipe);
        op.Process();
        return;
    }

    return;
}
