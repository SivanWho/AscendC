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
 * \file amp_update_scale.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"
#include "log/log.h"
#include "amp_update_scale_tiling.h"

namespace {
  constexpr uint64_t GROWTH_FACTOR_ID = 0;
  constexpr uint64_t BACKOFF_FACTOR_ID = 1;
  constexpr uint64_t GROWTH_INTERVAL = 2;
}

namespace optiling {

  class AmpUpdateScaleTiling {
    public:
        explicit AmpUpdateScaleTiling(gert::TilingContext* context) : TilingContext(context){};
        ge::graphStatus Init();
        ge::graphStatus RunKernelTiling();
        void TilingDataPrint() const;
    private:
        AmpUpdateScaleTilingData TilingData;
        gert::TilingContext* TilingContext = nullptr;
        float growthFactor = 0;
        float backoffFactor = 0;
        int32_t growthInterval = 0;
  };

  ge::graphStatus AmpUpdateScaleTiling::Init(){
    auto attrs = TilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, attrs);

    auto growthFactorPtr = attrs->GetAttrPointer<float>(GROWTH_FACTOR_ID);
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, growthFactorPtr);
    auto backoffFactorPtr = attrs->GetAttrPointer<float>(BACKOFF_FACTOR_ID);
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, backoffFactorPtr);
    auto growthIntervalPtr = attrs->GetAttrPointer<int64_t>(GROWTH_INTERVAL);
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, growthIntervalPtr);

    growthFactor = *growthFactorPtr;
    backoffFactor = *backoffFactorPtr;
    growthInterval = static_cast<int32_t>(*growthIntervalPtr);
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus AmpUpdateScaleTiling::RunKernelTiling(){
    TilingContext->SetBlockDim(1);

    auto currentScaleDtype = TilingContext->GetInputDesc(0)->GetDataType();
    uint64_t tilingKey = 0;
    if (currentScaleDtype == ge::DT_FLOAT) {
        tilingKey = 0;
    } else if (currentScaleDtype == ge::DT_FLOAT16) {
        tilingKey = 1;
    } else if (currentScaleDtype == ge::DT_BF16) {
        tilingKey = 2;
    }
    TilingContext->SetTilingKey(tilingKey);

    TilingData.set_growthFactor(growthFactor);
    TilingData.set_backoffFactor(backoffFactor);
    TilingData.set_growthInterval(growthInterval);

    TilingData.SaveToBuffer(TilingContext->GetRawTilingData()->GetData(), TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
  }

  void AmpUpdateScaleTiling::TilingDataPrint() const {
    OP_LOGI(TilingContext->GetNodeName(), "growthFactor:%f.", growthFactor);
    OP_LOGI(TilingContext->GetNodeName(), "backoffFactor:%f.", backoffFactor);
    OP_LOGI(TilingContext->GetNodeName(), "growthInterval:%d.", growthInterval);
  }

  static ge::graphStatus TilingAmpUpdateScale(gert::TilingContext* context) {
    AmpUpdateScaleTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
  }

  static ge::graphStatus TilingPrepareForAmpUpdateScale(gert::TilingParseContext* context) {
    OP_LOGD(context, "TilingPrepareForAmpUpdateScale start.");
    return ge::GRAPH_SUCCESS;
  }
  struct AmpUpdateScaleCompileInfo {};
  IMPL_OP_OPTILING(AmpUpdateScale)
      .Tiling(TilingAmpUpdateScale)
      .TilingParse<AmpUpdateScaleCompileInfo>(TilingPrepareForAmpUpdateScale);
}  // namespace optiling