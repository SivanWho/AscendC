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
 * \file complex_tiling.h
 * \brief Complex operator tiling header
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_COMPLEX_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_COMPLEX_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base_class.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "../../op_kernel/arch35/complex_struct.h"

namespace optiling {

struct ComplexCompileInfo {
    int64_t coreNum = 0;
};

class ComplexTiling : public Ops::Base::TilingBaseClass {
public:
    explicit ComplexTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    uint64_t GetTilingKey() const override { return 0; }
    ge::graphStatus GetWorkspaceSize() override { return ge::GRAPH_SUCCESS; }
    ge::graphStatus PostTiling() override;
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckBroadcastAndMergeShape();
    ge::graphStatus MergeDims();
    ge::graphStatus CalcStride();
    ge::graphStatus ComputeSimtConfigAndFill();
    void SearchSimtConfig(uint64_t N, int64_t C, int64_t& gridDim, int64_t& blockDim,
                          int64_t& elementsPerThread);
    void FillStridesAndMeta();

private:
    const ComplexCompileInfo* compileInfo_;
    ComplexTilingData tilingData_{};
    uint64_t totalElements_ = 0;
    int64_t dimNum_ = 0;
    uint64_t realDims_[COMPLEX_MAX_DIM] = {0};
    uint64_t imagDims_[COMPLEX_MAX_DIM] = {0};
    uint64_t mergedShape_[COMPLEX_MAX_DIM] = {0};
    uint64_t mergedStride_[COMPLEX_MAX_DIM] = {0};
    uint64_t realStride_[COMPLEX_MAX_DIM] = {0};
    uint64_t imagStride_[COMPLEX_MAX_DIM] = {0};
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_COMPLEX_TILING_H
