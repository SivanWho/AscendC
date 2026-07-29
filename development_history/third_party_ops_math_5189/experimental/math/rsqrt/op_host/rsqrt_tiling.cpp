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
 * \file rsqrt_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/rsqrt_tiling_data.h"
#include "../op_kernel/rsqrt_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
const uint32_t UB_DATA_NUM_BF16 = 9U;
const uint32_t UB_DATA_NUM_OTHER = 5U;

struct RsqrtCompileInfo {};

static ge::graphStatus TilingParseForRsqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RsqrtTilingFunc(gert::TilingContext* context)
{
    RsqrtTilingData* tiling = context->GetTilingData<RsqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RsqrtTilingData), 0, sizeof(RsqrtTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B &&
        context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        OP_LOGE(context, "socVersion error");
        return ge::GRAPH_FAILED;
    }

    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(inputNum == 0, OP_LOGE(context, "inputNum is 0"), return ge::GRAPH_FAILED);
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputBytes = inputLength / inputNum;

    uint64_t ubDataNumber =
        (context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) ? UB_DATA_NUM_BF16 : UB_DATA_NUM_OTHER;
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        coreNum = (coreNum < (int64_t)(inputLengthAlgin32 / BLOCK_SIZE)) ? coreNum :
                                                                           (int64_t)(inputLengthAlgin32 / BLOCK_SIZE);
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputBytes == 0, OP_LOGE(context, "inputBytes is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tileBlockNum == 0, OP_LOGE(context, "tileBlockNum is 0"), return ge::GRAPH_FAILED);

    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    tiling->smallCoreDataNum = (uint32_t)smallCoreDataNum;
    tiling->bigCoreDataNum = (uint32_t)bigCoreDataNum;
    tiling->tileDataNum = (uint32_t)tileDataNum;
    tiling->smallTailDataNum = (uint32_t)smallTailDataNum;
    tiling->bigTailDataNum = (uint32_t)bigTailDataNum;
    tiling->finalSmallTileNum = (uint32_t)finalSmallTileNum;
    tiling->finalBigTileNum = (uint32_t)finalBigTileNum;
    tiling->tailBlockNum = (uint32_t)tailBlockNum;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = sysWorkspaceSize;

    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Rsqrt).Tiling(RsqrtTilingFunc).TilingParse<RsqrtCompileInfo>(TilingParseForRsqrt);
} // namespace optiling
