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
 * \file clip_by_value_ub_broadcast_tiling.h
 * \brief
 */
#ifndef CLIP_BY_VALUE_UB_BROADCAST_TILING_H
#define CLIP_BY_VALUE_UB_BROADCAST_TILING_H

#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ClipByValueUbBrcTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubOuter);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubSplitAxis);
TILING_DATA_FIELD_DEF(int64_t, buffSize);
TILING_DATA_FIELD_DEF(int64_t, dimProductBeforeUbInner);
TILING_DATA_FIELD_DEF(int64_t, shapeLen);
TILING_DATA_FIELD_DEF(int64_t, runningRank);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input0Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input1Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input2Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, outputDims);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, outputStrides);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input0Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input1Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input2Strides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ClipByValue_3000001, ClipByValueUbBrcTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValue_3000002, ClipByValueUbBrcTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValue_3000003, ClipByValueUbBrcTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValue_3000004, ClipByValueUbBrcTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValue_3000009, ClipByValueUbBrcTilingData);

REGISTER_TILING_DATA_CLASS(ClipByValueV2_3000001, ClipByValueUbBrcTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValueV2_3000002, ClipByValueUbBrcTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValueV2_3000003, ClipByValueUbBrcTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValueV2_3000004, ClipByValueUbBrcTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValueV2_3000009, ClipByValueUbBrcTilingData);

class ClipByValueTilingUbBroadcast : public Ops::Base::TilingBaseClass {
public:
    explicit ClipByValueTilingUbBroadcast(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    uint64_t GetOpKey(
        ge::DataType xDataType, ge::DataType clipValueMinDtype, ge::DataType clipValueMaxDtype, ge::DataType yDataType);
    std::map<uint64_t, Ops::Base::BroadcastComputeParams> GetComputeMap(uint64_t keyData);

    ge::graphStatus DoBroadcastTiling(
        const Ops::Base::BroadcastTilingParams& broadcastTilingParams,
        Ops::Base::BroadcastTilingData& broadcastTilingData);

    ge::graphStatus GetDtypes();

    ge::graphStatus DoDimensionCollapse();

    ge::graphStatus SetTilingData(Ops::Base::BroadcastTilingData& broadcastTilingData);

    ge::graphStatus FindComputeParams(const Ops::Base::BroadcastTilingParams& params,
                                       Ops::Base::BroadcastComputeParams& computeParams);
    ge::graphStatus CalcUbSplitParams(const Ops::Base::BroadcastComputeParams& computeParams,
                                       uint64_t& ubSplitAxes, uint64_t& curProduct);
    ge::graphStatus CalcUbTailParams(uint64_t ubSplitAxes, uint64_t curProduct, uint64_t maxElemNum,
                                     uint32_t& ubFormer, uint64_t& ubOuter, uint64_t& ubTail);
    ge::graphStatus CalcBlockSplitParams(uint64_t ubSplitAxes, uint64_t ubOuter,
                               const Ops::Base::BroadcastTilingParams& params,
                               Ops::Base::BroadcastTilingData& data);

    void CopyAndSetDimsStrides(const Ops::Base::BroadcastTilingData& data);

    const char* opName = "ClipByValue";
    int64_t coreNum{0};    // syscfg
    uint64_t ubSize{0};    // syscfg
    int64_t opKey{0};
    int64_t buffSize{32};
    int64_t blockNum{0};
    int64_t input0Dims[MAX_DIM_SIZE] = {0};
    int64_t input1Dims[MAX_DIM_SIZE] = {0};
    int64_t input2Dims[MAX_DIM_SIZE] = {0};
    int64_t outputDims[MAX_DIM_SIZE] = {0};
    int64_t outputStrides[MAX_DIM_SIZE] = {0};
    int64_t input0Strides[MAX_DIM_SIZE] = {0};
    int64_t input1Strides[MAX_DIM_SIZE] = {0};
    int64_t input2Strides[MAX_DIM_SIZE] = {0};

    ge::DataType xDtype{ge::DataType::DT_FLOAT};
    ge::DataType minDtype{ge::DataType::DT_FLOAT};
    ge::DataType maxDtype{ge::DataType::DT_FLOAT};
    ge::DataType yDtype{ge::DataType::DT_FLOAT};

    uint32_t dTypeSize{1};
    bool isUbBroadcast{false};

    std::vector<std::vector<int64_t>> dims;
    std::vector<std::vector<int64_t>> strides;

    ClipByValueUbBrcTilingData tilingData;
};
} // namespace optiling
#endif