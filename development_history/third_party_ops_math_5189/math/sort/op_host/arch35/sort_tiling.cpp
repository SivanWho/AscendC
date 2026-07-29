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
 * \file sort_tiling.cpp
 * \brief
 */
#include "sort_tiling.h"
#include "sort_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

using namespace ge;

namespace {
const int64_t INT_BYTES = 4;
const int64_t SORT_BYTES = 8;  // for 910B sort struct
const int64_t TMP_VAL = 0;
const int64_t TMP_IDX = 1;
const int64_t TMP_CACHE = 2;
const int64_t TMP_CONV = 3;
const string kSortWithIndex = "SortWithIndex";
}  // namespace

namespace optiling {
static ge::graphStatus Tiling4Sort(gert::TilingContext* context) {
  auto platformInfo = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo); 
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  std::string opType(context->GetNodeType());
  OP_LOGD(context->GetNodeName(), "AscendC Sort simt tiling");
  OP_CHECK_IF(SortTilingSimt(context, ascendcPlatform.GetCoreNumAiv()) != ge::GRAPH_SUCCESS, 
                  OP_LOGE(context->GetNodeName(),
                  "The simt tiling function failed"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Sort(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "AscendC Tiling starting GRAPH_SUCCESS");
  auto platformInfo = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo); 
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  OP_CHECK_IF((ascendcPlatform.GetCoreNumAiv() <= 0),
                  OP_LOGE(context->GetNodeName(),
                  "The core num is invaild."),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sort).Tiling(Tiling4Sort).TilingParse<SortCompileInfo>(TilingPrepare4Sort);
IMPL_OP_OPTILING(SortV2).Tiling(Tiling4Sort).TilingParse<SortCompileInfo>(TilingPrepare4Sort);
}  // namespace optiling