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
 * \file clip_by_value_tiling.cpp
 * \brief
 */

#include "clip_by_value_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr uint64_t CLIP_BY_VALUE_COMMON_TILING_PRIORITY = 3;

static constexpr uint64_t OP_KEY_INVALID = 0;
static constexpr uint64_t OP_KEY_1 = 1;
static constexpr uint64_t OP_KEY_2 = 2;
static constexpr uint64_t OP_KEY_3 = 3;
static constexpr uint64_t OP_KEY_4 = 4;
static constexpr uint64_t OP_KEY_5 = 5;
static constexpr uint64_t INDEX_0 = 0;
static constexpr uint64_t INDEX_1 = 1;
static constexpr uint64_t INDEX_2 = 2;
static constexpr uint64_t INDEX_3 = 3;
static constexpr uint64_t WORKSPACE_SIZE = 32;
static constexpr int64_t BUFFER_DIVISOR_BITS16 = 128;
static constexpr int64_t BUFFER_DIVISOR_BITS32 = 256;
static constexpr int64_t BUFFER_DIVISOR_BITS64 = 512;

ge::graphStatus ClipByValueTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<ClipByValueCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum = static_cast<int64_t>(compileInfoPtr->coreNum);
        ubSize = static_cast<int64_t>(compileInfoPtr->ubSize);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = static_cast<int64_t>(ubSizePlatForm);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t ClipByValueTiling::GetOpKey(
    ge::DataType xDtype, ge::DataType clipValueMinDtype, ge::DataType clipValueMaxDtype, ge::DataType yDtype)
{
    bool opKey1Flag = xDtype == DT_FLOAT16 && clipValueMinDtype == DT_FLOAT16 && clipValueMaxDtype == DT_FLOAT16 &&
                      yDtype == DT_FLOAT16;
    if (opKey1Flag) {
        return OP_KEY_1;
    }
    bool opKey2Flag =
        xDtype == DT_FLOAT && clipValueMinDtype == DT_FLOAT && clipValueMaxDtype == DT_FLOAT && yDtype == DT_FLOAT;
    if (opKey2Flag) {
        return OP_KEY_2;
    }
    bool opKey3Flag =
        xDtype == DT_BF16 && clipValueMinDtype == DT_BF16 && clipValueMaxDtype == DT_BF16 && yDtype == DT_BF16;
    if (opKey3Flag) {
        return OP_KEY_3;
    }
    bool opKey4Flag =
        xDtype == DT_INT32 && clipValueMinDtype == DT_INT32 && clipValueMaxDtype == DT_INT32 && yDtype == DT_INT32;
    if (opKey4Flag) {
        return OP_KEY_4;
    }
    bool opKey5Flag =
        xDtype == DT_INT64 && clipValueMinDtype == DT_INT64 && clipValueMaxDtype == DT_INT64 && yDtype == DT_INT64;
    if (opKey5Flag) {
        return OP_KEY_5;
    }

    return OP_KEY_INVALID;
}

uint64_t ClipByValueTiling::GenerateTilingKey(uint64_t innerKey)
{
    return opKey * Ops::Base::BROADCAST_OP_KEY_OFFSET + innerKey;
}

std::map<uint64_t, Ops::Base::BroadcastComputeParams> ClipByValueTiling::GetComputeMap(uint64_t opKey)
{
    Ops::Base::BroadcastComputeParams computeParams0;
    switch (opKey) {
        case OP_KEY_1:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {BUFFER_DIVISOR_BITS16, BUFFER_DIVISOR_BITS16};
            return {{1, computeParams0}};
        case OP_KEY_2:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {BUFFER_DIVISOR_BITS32, BUFFER_DIVISOR_BITS32};
            return {{1, computeParams0}};
        case OP_KEY_3:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {BUFFER_DIVISOR_BITS16, BUFFER_DIVISOR_BITS16};
            return {{1, computeParams0}};
        case OP_KEY_4:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {BUFFER_DIVISOR_BITS32, BUFFER_DIVISOR_BITS32};
            return {{1, computeParams0}};
        case OP_KEY_5:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS64_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS64_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {BUFFER_DIVISOR_BITS64, BUFFER_DIVISOR_BITS64};
            return {{1, computeParams0}};
        default:
            return {};
    }
}

ge::graphStatus ClipByValueTiling::GetShapeAttrsInfo()
{
    auto x = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    auto xDtype = x->GetDataType();

    auto clipValueMin = context_->GetInputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, clipValueMin);
    auto clipValueMinDtype = clipValueMin->GetDataType();

    auto clipValueMax = context_->GetInputDesc(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, clipValueMax);
    auto clipValueMaxDtype = clipValueMax->GetDataType();

    auto y = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y);
    auto yDtype = y->GetDataType();

    opKey = GetOpKey(xDtype, clipValueMinDtype, clipValueMaxDtype, yDtype);
    if (opKey == OP_KEY_INVALID) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(clipValueMinDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(clipValueMaxDtype) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(yDtype);
        std::string reasonMsg =
            "The dtype of input x must be within the range of FLOAT, FLOAT16, BFLOAT16, INT32, or INT64, "
            "and it's dtype must be the same as the dtypes of clip_value_min, clip_value_max, and y";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x, clip_value_min, clip_value_max and y", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool ClipByValueTiling::IsCapable()
{
    return true;
}

ge::graphStatus ClipByValueTiling::DoOpTiling()
{
    Ops::Base::BroadcastTilingParams broadcastTilingParams;
    for (uint64_t i = 0; i < context_->GetComputeNodeInputNum(); i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        broadcastTilingParams.inShape.push_back(Ops::Base::EnsureNotScalar(shape->GetStorageShape()));
    }

    auto outputShape = context_->GetOutputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    broadcastTilingParams.outShape = Ops::Base::EnsureNotScalar(outputShape->GetStorageShape());
    broadcastTilingParams.computeMap = GetComputeMap(opKey);
    broadcastTilingParams.coreNum = coreNum;
    broadcastTilingParams.ubSize = ubSize;

    Ops::Base::BroadcastTilingData broadcastTilingData;
    ge::graphStatus status = BroadcastTiling(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "broadcast tiling failed.");
        return ge::GRAPH_FAILED;
    }

    tilingKey_ = GenerateTilingKey(broadcastTilingData.innerKey);
    blockNum = broadcastTilingData.blockNum;
    tilingData.set_blockFormer(broadcastTilingData.blockFormer);
    tilingData.set_ubFormer(broadcastTilingData.ubFormer);
    tilingData.set_ubOuter(broadcastTilingData.ubOuter);
    tilingData.set_ubTail(broadcastTilingData.ubTail);
    tilingData.set_blockTail(broadcastTilingData.blockTail);
    tilingData.set_shapeLen(broadcastTilingData.shapeLen);
    tilingData.set_ubSplitAxis(broadcastTilingData.ubSplitAxis);
    tilingData.set_dimProductBeforeUbInner(broadcastTilingData.dimProductBeforeUbInner);
    tilingData.set_elemNum(broadcastTilingData.elemNum);

    std::copy(broadcastTilingData.dims[INDEX_0].begin(), broadcastTilingData.dims[INDEX_0].end(), input0Dims);
    tilingData.set_input0Dims(input0Dims);
    std::copy(broadcastTilingData.dims[INDEX_1].begin(), broadcastTilingData.dims[INDEX_1].end(), input1Dims);
    tilingData.set_input1Dims(input1Dims);
    std::copy(broadcastTilingData.dims[INDEX_2].begin(), broadcastTilingData.dims[INDEX_2].end(), input2Dims);
    tilingData.set_input2Dims(input2Dims);
    std::copy(broadcastTilingData.dims[INDEX_3].begin(), broadcastTilingData.dims[INDEX_3].end(), outputDims);
    tilingData.set_outputDims(outputDims);
    std::copy(broadcastTilingData.strides[INDEX_3].begin(), broadcastTilingData.strides[INDEX_3].end(), outputStrides);
    tilingData.set_outputStrides(outputStrides);
    std::copy(broadcastTilingData.strides[INDEX_0].begin(), broadcastTilingData.strides[INDEX_0].end(), input0Strides);
    tilingData.set_input0Strides(input0Strides);
    std::copy(broadcastTilingData.strides[INDEX_1].begin(), broadcastTilingData.strides[INDEX_1].end(), input1Strides);
    tilingData.set_input1Strides(input1Strides);
    std::copy(broadcastTilingData.strides[INDEX_2].begin(), broadcastTilingData.strides[INDEX_2].end(), input2Strides);
    tilingData.set_input2Strides(input2Strides);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ClipByValueTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus ClipByValueTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    auto ToString = [](ClipByValueTilingData& localTilingData) -> std::string {
        std::string str;
        str += " blockFormer:" + std::to_string(localTilingData.get_blockFormer());
        str += " ubFormer:" + std::to_string(localTilingData.get_ubFormer());
        str += " ubOuter:" + std::to_string(localTilingData.get_ubOuter());
        str += " ubTail:" + std::to_string(localTilingData.get_ubTail());
        str += " blockTail:" + std::to_string(localTilingData.get_blockTail());
        str += " shapeLen:" + std::to_string(localTilingData.get_shapeLen());
        str += " ubSplitAxis:" + std::to_string(localTilingData.get_ubSplitAxis());
        str += " dimProductBeforeUbInner:" + std::to_string(localTilingData.get_dimProductBeforeUbInner());
        str += " elemNum:" + std::to_string(localTilingData.get_elemNum());
        return str;
    };

    OP_LOGI(context_, "TilingInfo: %s.", ToString(tilingData).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForClipByValue(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingForClipByValue", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = context->GetCompileInfo<ClipByValueCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForClipByValue(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForClipByValue", "TilingContext is nullptr.");
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

IMPL_OP_OPTILING(ClipByValue)
    .Tiling(TilingForClipByValue)
    .TilingParse<ClipByValueCompileInfo>(TilingPrepareForClipByValue);

REGISTER_OPS_TILING_TEMPLATE(ClipByValue, ClipByValueTiling, CLIP_BY_VALUE_COMMON_TILING_PRIORITY);

} // namespace optiling
