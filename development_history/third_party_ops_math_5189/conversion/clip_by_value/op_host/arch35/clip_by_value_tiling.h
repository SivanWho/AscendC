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
 * \file clip_by_value_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_CLIP_BY_VALUE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_CLIP_BY_VALUE_H_

#include "atvoss/broadcast/broadcast_tiling.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"

namespace optiling {
static constexpr int64_t MAX_DIM_SIZE = 8;

BEGIN_TILING_DATA_DEF(ClipByValueTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubOuter);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, shapeLen);
TILING_DATA_FIELD_DEF(int64_t, ubSplitAxis);
TILING_DATA_FIELD_DEF(int64_t, dimProductBeforeUbInner);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input0Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input1Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input2Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, outputDims);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, outputStrides);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input0Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input1Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_SIZE, input2Strides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ClipByValue, ClipByValueTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValueV2, ClipByValueTilingData);

struct ClipByValueCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class ClipByValueTiling : public Ops::Base::TilingBaseClass {
public:
    explicit ClipByValueTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    static uint64_t GetOpKey(
        ge::DataType xDtype, ge::DataType clipValueMinDtype, ge::DataType clipValueMaxDtype, ge::DataType yDtype);
    uint64_t GenerateTilingKey(uint64_t innerKey);
    static std::map<uint64_t, Ops::Base::BroadcastComputeParams> GetComputeMap(uint64_t opKey);

    ClipByValueTilingData tilingData;
    uint64_t opKey{0};
    int64_t coreNum{0};
    int64_t ubSize{0};
    uint64_t blockNum{0};
    int64_t input0Dims[MAX_DIM_SIZE] = {0};
    int64_t input1Dims[MAX_DIM_SIZE] = {0};
    int64_t input2Dims[MAX_DIM_SIZE] = {0};
    int64_t outputDims[MAX_DIM_SIZE] = {0};
    int64_t outputStrides[MAX_DIM_SIZE] = {0};
    int64_t input0Strides[MAX_DIM_SIZE] = {0};
    int64_t input1Strides[MAX_DIM_SIZE] = {0};
    int64_t input2Strides[MAX_DIM_SIZE] = {0};
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_CLIP_BY_VALUE_H_
