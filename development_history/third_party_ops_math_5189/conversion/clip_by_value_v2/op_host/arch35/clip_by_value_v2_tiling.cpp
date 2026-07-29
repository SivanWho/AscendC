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
 * \file clip_by_value_v2_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "conversion/clip_by_value/op_host/arch35/clip_by_value_tiling.h"
#include "platform/platform_info.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr uint64_t CLIP_BY_VALUE_COMMON_TILING_PRIORITY = 3;

static ge::graphStatus TilingForClipByValueV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingForClipByValueV2", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = context->GetCompileInfo<ClipByValueCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForClipByValueV2(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForClipByValueV2", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfoPtr = context->GetCompiledInfo<ClipByValueCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ClipByValueV2)
    .Tiling(TilingForClipByValueV2)
    .TilingParse<ClipByValueCompileInfo>(TilingPrepareForClipByValueV2);

REGISTER_OPS_TILING_TEMPLATE(ClipByValueV2, ClipByValueTiling, CLIP_BY_VALUE_COMMON_TILING_PRIORITY);

} // namespace optiling
