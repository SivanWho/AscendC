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
 * \file clip_by_value_ub_broadcast_tiling.cpp
 * \brief
 */
#include <vector>
#include "log/log.h"
#include "clip_by_value_tiling.h"
#include "clip_by_value_ub_broadcast_tiling.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace ge;
using namespace std;
namespace ClipByValueUbBroadcast {
constexpr int64_t INOUT_PARAM_NUM = 4;
constexpr int64_t X_INDEX = 0;
constexpr int64_t MIN_INDEX = 1;
constexpr int64_t MAX_INDEX = 2;
constexpr int64_t Y_INDEX = 3;
constexpr int64_t OUTPUT_INDEX = 0;

constexpr int64_t MAX_RANK_PATTERN = 4;
constexpr int64_t PATTERN_RANK_UNKNOWN = 9;
constexpr uint64_t UB_BROADCAST_OP_KEY_OFFSET = 3000000;

constexpr int32_t UB_BROADCAST_NODE_NUM = 11;
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
constexpr int64_t UB_BROADCAST_TILE_COUNT = 87;
constexpr int64_t UB_BROADCAST_TILE_SIZE = 128;

constexpr uint64_t OP_KEY_INVALID = 0;
constexpr uint64_t OP_KEY_1 = 1;
constexpr uint64_t OP_KEY_2 = 2;
constexpr uint64_t OP_KEY_3 = 3;
constexpr uint64_t OP_KEY_4 = 4;
constexpr uint64_t OP_KEY_5 = 5;
constexpr uint64_t INDEX_0 = 0;
constexpr uint64_t INDEX_1 = 1;
constexpr uint64_t INDEX_2 = 2;
constexpr uint64_t INDEX_3 = 3;
} // namespace ClipByValueUbBroadcast

namespace optiling {

static constexpr uint64_t CLIP_BY_VALUE_UB_BROADCAST_TILING_PRIORITY = 2;
static constexpr int64_t UB_ALIGN_SIZE = 32;

static string ClipByValueTensorUbBrcDesc2String(
    const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << "),";
    oss << "(shape:" << Ops::Base::ToString(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << Ops::Base::ToString(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToSerialString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToSerialString(tensor->GetOriginFormat()) << ") ";

    return oss.str();
}

static string ClipByValueUbBrcDebugTilingContext(gert::TilingContext* context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << ClipByValueTensorUbBrcDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << ClipByValueTensorUbBrcDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

bool ClipByValueTilingUbBroadcast::IsCapable()
{
    return isUbBroadcast;
}

ge::graphStatus ClipByValueTilingUbBroadcast::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<ClipByValueCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    opName = context_->GetNodeName();
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from compile info.");
        coreNum = static_cast<int64_t>(compileInfo->coreNum);
        ubSize = static_cast<int64_t>(compileInfo->ubSize);
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from platform.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizeTemp = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
        ubSize = static_cast<int64_t>(ubSizeTemp);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t ClipByValueTilingUbBroadcast::GetOpKey(
    ge::DataType xDataType, ge::DataType clipValueMinDtype, ge::DataType clipValueMaxDtype, ge::DataType yDataType)
{
    bool opKey1Flag = xDataType == DT_FLOAT16 && clipValueMinDtype == DT_FLOAT16 && clipValueMaxDtype == DT_FLOAT16 &&
                      yDataType == DT_FLOAT16;
    if (opKey1Flag) {
        return ClipByValueUbBroadcast::OP_KEY_1;
    }
    bool opKey2Flag =
        xDataType == DT_FLOAT && clipValueMinDtype == DT_FLOAT && clipValueMaxDtype == DT_FLOAT && yDataType == DT_FLOAT;
    if (opKey2Flag) {
        return ClipByValueUbBroadcast::OP_KEY_2;
    }
    bool opKey3Flag =
        xDataType == DT_BF16 && clipValueMinDtype == DT_BF16 && clipValueMaxDtype == DT_BF16 && yDataType == DT_BF16;
    if (opKey3Flag) {
        return ClipByValueUbBroadcast::OP_KEY_3;
    }
    bool opKey4Flag =
        xDataType == DT_INT32 && clipValueMinDtype == DT_INT32 && clipValueMaxDtype == DT_INT32 && yDataType == DT_INT32;
    if (opKey4Flag) {
        return ClipByValueUbBroadcast::OP_KEY_4;
    }
    bool opKey5Flag =
        xDataType == DT_INT64 && clipValueMinDtype == DT_INT64 && clipValueMaxDtype == DT_INT64 && yDataType == DT_INT64;
    if (opKey5Flag) {
        return ClipByValueUbBroadcast::OP_KEY_5;
    }

    return ClipByValueUbBroadcast::OP_KEY_INVALID;
}

ge::graphStatus ClipByValueTilingUbBroadcast::DoDimensionCollapse()
{
    auto xShape = context_->GetInputShape(ClipByValueUbBroadcast::X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = Ops::Base::EnsureNotScalar(xShape->GetStorageShape());

    auto minShape = context_->GetInputShape(ClipByValueUbBroadcast::MIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minShape);
    auto minStorageShape = Ops::Base::EnsureNotScalar(minShape->GetStorageShape());

    auto maxShape = context_->GetInputShape(ClipByValueUbBroadcast::MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxShape);
    auto maxStorageShape = Ops::Base::EnsureNotScalar(maxShape->GetStorageShape());

    auto yShape = context_->GetOutputShape(ClipByValueUbBroadcast::OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = Ops::Base::EnsureNotScalar(yShape->GetStorageShape());

    std::vector<gert::Shape> inShapes;
    inShapes.push_back(xStorageShape);
    inShapes.push_back(minStorageShape);
    inShapes.push_back(maxStorageShape);

    ge::graphStatus res = Ops::Base::DimensionCollapse(inShapes, yStorageShape, dims, strides);

    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "DimensionCollapse failed."), return res);

    if (dims.size() != ClipByValueUbBroadcast::INOUT_PARAM_NUM) {
        OP_LOGE(context_->GetNodeName(),
            "DimensionCollapse failed. Check dims is illegal, out dims num %lu not equal to %ld, please check the "
            "shapes of x, clip_value_min, clip_value_max and y %s %s %s and %s",
            dims.size(), ClipByValueUbBroadcast::INOUT_PARAM_NUM, Ops::Base::ToString(xStorageShape).c_str(),
            Ops::Base::ToString(minStorageShape).c_str(), Ops::Base::ToString(maxStorageShape).c_str(),
            Ops::Base::ToString(yStorageShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (static_cast<int64_t>(dims.back().size()) > static_cast<int64_t>(Ops::Base::BROADCAST_MAX_DIMS)) {
        OP_LOGE(context_->GetNodeName(),
            "DimensionCollapse failed. Broadcast can't support dim size greater than 8, please check the "
            "shapes of x, clip_value_min, clip_value_max and y %s %s %s and %s",
            Ops::Base::ToString(xStorageShape).c_str(), Ops::Base::ToString(minStorageShape).c_str(),
            Ops::Base::ToString(maxStorageShape).c_str(), Ops::Base::ToString(yStorageShape).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::GetDtypes()
{
    auto xDesc = context_->GetInputDesc(ClipByValueUbBroadcast::X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype = xDesc->GetDataType();

    auto minDesc = context_->GetInputDesc(ClipByValueUbBroadcast::MIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minDesc);
    minDtype = minDesc->GetDataType();

    auto maxDesc = context_->GetInputDesc(ClipByValueUbBroadcast::MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxDesc);
    maxDtype = maxDesc->GetDataType();

    auto yDesc = context_->GetOutputDesc(ClipByValueUbBroadcast::OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    yDtype = yDesc->GetDataType();
    if (xDtype != minDtype || xDtype != maxDtype || xDtype != yDtype) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(minDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(maxDtype) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(yDtype);
        std::string reasonMsg = "Dtypes of x, clip_value_min, clip_value_max and y must be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x, clip_value_min, clip_value_max and y", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_INT32 && xDtype != ge::DataType::DT_FLOAT16 &&
        xDtype != ge::DataType::DT_BF16 && xDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
            "float, float16, bfloat16, int32 and int64");
        return ge::GRAPH_FAILED;
    }

    dTypeSize = ge::GetSizeByDataType(xDtype);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::GetShapeAttrsInfo()
{
    OP_LOGD(
        context_->GetNodeName(), "TilingContext: %s", optiling::ClipByValueUbBrcDebugTilingContext(context_).c_str());

    ge::graphStatus res = ClipByValueTilingUbBroadcast::GetDtypes();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    res = DoDimensionCollapse();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    opKey = GetOpKey(xDtype, minDtype, maxDtype, yDtype);
    if (opKey == ClipByValueUbBroadcast::OP_KEY_INVALID) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(minDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(maxDtype) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(yDtype);
        std::string reasonMsg =
            "The dtype of input x must be within the range of FLOAT, FLOAT16, BFLOAT16, INT32, or INT64, "
            "and it's dtype must be the same as the dtypes of clip_value_min, clip_value_max, and y";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x, clip_value_min, clip_value_max and y", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // UB内broadcast条件： 尾轴32B对齐场景。
    int64_t yLastDim = dims[ClipByValueUbBroadcast::Y_INDEX].back();
    // ub broadcast 高阶接口暂时只支持-1轴和-2轴
    int64_t yElementNum = dims[ClipByValueUbBroadcast::Y_INDEX].size();
    int64_t yPenultimateDim = yElementNum > 1 ? dims[ClipByValueUbBroadcast::Y_INDEX][yElementNum - 2] : 1;
    constexpr int64_t UB_BROADCAST_SUPPORT_LEN = ClipByValueUbBroadcast::UB_BROADCAST_TILE_COUNT * ClipByValueUbBroadcast::UB_BROADCAST_TILE_SIZE;

    if ((yLastDim * dTypeSize % UB_ALIGN_SIZE == 0) && (xDtype == DT_FLOAT16 || xDtype == DT_BF16) &&
        (yLastDim > UB_BROADCAST_SUPPORT_LEN || yLastDim * yPenultimateDim > UB_BROADCAST_SUPPORT_LEN)) {
        isUbBroadcast = true;
    }

    OP_LOGI(
        context_->GetNodeName(), "isUbBroadcast: %d, yLastDim: %ld, yElements: %ld, yPenultimateDim: %ld ",
        isUbBroadcast, yLastDim, yElementNum, yPenultimateDim);

    return ge::GRAPH_SUCCESS;
}

std::map<uint64_t, Ops::Base::BroadcastComputeParams> ClipByValueTilingUbBroadcast::GetComputeMap(uint64_t keyData)
{
    Ops::Base::BroadcastComputeParams computeParams0;
    buffSize = (ubSize / ClipByValueUbBroadcast::UB_BROADCAST_NODE_NUM / dTypeSize) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
    // maxElemNums indicates the maximum number of elements in ub, which is calculated at compile time.
    switch (keyData) {
        case ClipByValueUbBroadcast::OP_KEY_1:
            computeParams0.maxDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {buffSize, buffSize};
            return {{1, computeParams0}};
        case ClipByValueUbBroadcast::OP_KEY_2:
            computeParams0.maxDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {buffSize, buffSize};
            return {{1, computeParams0}};
        case ClipByValueUbBroadcast::OP_KEY_3:
            computeParams0.maxDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {buffSize, buffSize};
            return {{1, computeParams0}};
        case ClipByValueUbBroadcast::OP_KEY_4:
            computeParams0.maxDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {buffSize, buffSize};
            return {{1, computeParams0}};
        case ClipByValueUbBroadcast::OP_KEY_5:
            computeParams0.maxDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS64_SIZE);
            computeParams0.minDtypeBits = static_cast<uint64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS64_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {buffSize, buffSize};
            return {{1, computeParams0}};
        default:
            return {};
    }
}

void ClipByValueTilingUbBroadcast::CopyAndSetDimsStrides(const Ops::Base::BroadcastTilingData& data)
{
    std::copy(data.dims[ClipByValueUbBroadcast::INDEX_0].begin(),
              data.dims[ClipByValueUbBroadcast::INDEX_0].end(), input0Dims);
    tilingData.set_input0Dims(input0Dims);
    std::copy(data.dims[ClipByValueUbBroadcast::INDEX_1].begin(),
              data.dims[ClipByValueUbBroadcast::INDEX_1].end(), input1Dims);
    tilingData.set_input1Dims(input1Dims);
    std::copy(data.dims[ClipByValueUbBroadcast::INDEX_2].begin(),
              data.dims[ClipByValueUbBroadcast::INDEX_2].end(), input2Dims);
    tilingData.set_input2Dims(input2Dims);
    std::copy(data.dims[ClipByValueUbBroadcast::INDEX_3].begin(),
              data.dims[ClipByValueUbBroadcast::INDEX_3].end(), outputDims);
    tilingData.set_outputDims(outputDims);

    std::copy(data.strides[ClipByValueUbBroadcast::INDEX_3].begin(),
              data.strides[ClipByValueUbBroadcast::INDEX_3].end(), outputStrides);
    tilingData.set_outputStrides(outputStrides);
    std::copy(data.strides[ClipByValueUbBroadcast::INDEX_0].begin(),
              data.strides[ClipByValueUbBroadcast::INDEX_0].end(), input0Strides);
    tilingData.set_input0Strides(input0Strides);
    std::copy(data.strides[ClipByValueUbBroadcast::INDEX_1].begin(),
              data.strides[ClipByValueUbBroadcast::INDEX_1].end(), input1Strides);
    tilingData.set_input1Strides(input1Strides);
    std::copy(data.strides[ClipByValueUbBroadcast::INDEX_2].begin(),
              data.strides[ClipByValueUbBroadcast::INDEX_2].end(), input2Strides);
    tilingData.set_input2Strides(input2Strides);
}

ge::graphStatus ClipByValueTilingUbBroadcast::SetTilingData(Ops::Base::BroadcastTilingData& broadcastTilingData)
{
    blockNum = broadcastTilingData.blockNum;
    tilingData.set_blockFormer(broadcastTilingData.blockFormer);
    tilingData.set_ubFormer(broadcastTilingData.ubFormer);
    tilingData.set_ubOuter(broadcastTilingData.ubOuter);
    tilingData.set_ubTail(broadcastTilingData.ubTail);
    tilingData.set_blockTail(broadcastTilingData.blockTail);
    tilingData.set_dimProductBeforeUbInner(broadcastTilingData.dimProductBeforeUbInner);
    tilingData.set_shapeLen(broadcastTilingData.shapeLen);
    tilingData.set_ubSplitAxis(broadcastTilingData.ubSplitAxis);
    tilingData.set_buffSize(buffSize);

    int64_t runningRank = tilingData.get_shapeLen() - tilingData.get_ubSplitAxis();
    int64_t rank = runningRank > ClipByValueUbBroadcast::MAX_RANK_PATTERN ?
                       ClipByValueUbBroadcast::PATTERN_RANK_UNKNOWN : runningRank;
    tilingKey_ = ClipByValueUbBroadcast::UB_BROADCAST_OP_KEY_OFFSET + rank;
    tilingData.set_runningRank(runningRank);

    CopyAndSetDimsStrides(broadcastTilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::DoOpTiling()
{
    Ops::Base::BroadcastTilingParams broadcastTilingParams;
    for (uint64_t i = 0; i < context_->GetComputeNodeInputNum(); i++) {
        auto inputShape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
        broadcastTilingParams.inShape.push_back(
            Ops::Base::EnsureNotScalar(inputShape->GetStorageShape()));
    }

    auto outputShape = context_->GetOutputShape(ClipByValueUbBroadcast::OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    broadcastTilingParams.outShape = Ops::Base::EnsureNotScalar(outputShape->GetStorageShape());
    
    OP_CHECK_IF(dTypeSize == 0,
                OP_LOGE("DoOpTiling", "dTypeSize can not be 0"),
                return ge::GRAPH_FAILED);
    
    broadcastTilingParams.computeMap = GetComputeMap(opKey);
    broadcastTilingParams.coreNum = coreNum;
    broadcastTilingParams.ubSize = ubSize;

    Ops::Base::BroadcastTilingData broadcastTilingData;

    ge::graphStatus res = DoBroadcastTiling(broadcastTilingParams, broadcastTilingData);
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
            OP_LOGE("BroadcastTiling", "DoBroadcastTiling failed."),
            return ge::GRAPH_FAILED);

    SetTilingData(broadcastTilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ClipByValueTilingUbBroadcast::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus ClipByValueTilingUbBroadcast::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = ClipByValueUbBroadcast::MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(blockNum);

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

// 覆盖BroadcastTiling函数
static constexpr uint64_t BROADCAST_COMPUTE_KEY = 1;

ge::graphStatus ClipByValueTilingUbBroadcast::FindComputeParams(
    const Ops::Base::BroadcastTilingParams& params, Ops::Base::BroadcastComputeParams& computeParams)
{
    auto iter = params.computeMap.find(BROADCAST_COMPUTE_KEY);
    if (iter != params.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("BroadcastTiling", "can not find computeKey");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(params.ubSize < computeParams.extraSize[0],
                OP_LOGE("BroadcastTiling", "ubSize is smaller than extra size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::CalcUbSplitParams(
    const Ops::Base::BroadcastComputeParams& computeParams, uint64_t& ubSplitAxes, uint64_t& curProduct)
{
    uint64_t maxElemNum = computeParams.bufferDivisor[0];
    OP_CHECK_IF(maxElemNum == 0,
                OP_LOGE("BroadcastTiling", "maxElemNum can not be 0"),
                return ge::GRAPH_FAILED);

    curProduct = 1;
    ubSplitAxes = 0;
    bool flag = true;
    for (int64_t i = dims.back().size() - 1; i >= 0; i--) {
        curProduct *= dims.back()[i];
        if (curProduct > maxElemNum) {
            curProduct = curProduct / dims.back()[i];
            ubSplitAxes = i;
            flag = false;
            break;
        }
    }
    if (flag) {
        curProduct = curProduct / dims.back()[0];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::CalcUbTailParams(uint64_t ubSplitAxes, uint64_t curProduct,
                                                               uint64_t maxElemNum, uint32_t& ubFormer,
                                                               uint64_t& ubOuter, uint64_t& ubTail)
{
    if (dims.back().size() == 1) {
        ubFormer = maxElemNum;
    } else {
        OP_CHECK_IF(curProduct == 0,
                    OP_LOGE("BroadcastTiling", "curProduct can not be 0"),
                    return ge::GRAPH_FAILED);
        ubFormer = maxElemNum / curProduct;
    }
    OP_CHECK_IF(ubFormer == 0,
                OP_LOGE("BroadcastTiling", "ubFormer can not be 0"),
                return ge::GRAPH_FAILED);
    ubOuter = (dims.back()[ubSplitAxes] + ubFormer - 1) / ubFormer;
    ubTail = dims.back()[ubSplitAxes] - (ubOuter - 1) * ubFormer;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::CalcBlockSplitParams(uint64_t ubSplitAxes, uint64_t ubOuter,
                                                                 const Ops::Base::BroadcastTilingParams& params,
                                                                 Ops::Base::BroadcastTilingData& data)
{
    uint64_t fusedProduct = ubOuter;
    for (uint64_t i = 0; i < ubSplitAxes; i++) {
        fusedProduct *= dims.back()[i];
    }

    OP_CHECK_IF(params.coreNum == 0,
                OP_LOGE("BroadcastTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);

    uint64_t blockFormer = (fusedProduct + params.coreNum - 1) / params.coreNum;
    OP_CHECK_IF(blockFormer == 0,
                OP_LOGE("BroadcastTiling", "blockFormer can not be 0, fusedProduct may be 0"),
                return ge::GRAPH_FAILED);

    uint64_t localBlockNum = (fusedProduct + blockFormer - 1) / blockFormer;
    uint64_t blockTail = fusedProduct - (localBlockNum - 1) * blockFormer;

    data.blockFormer = blockFormer;
    data.blockNum = localBlockNum;
    data.blockTail = blockTail;
    data.dimProductBeforeUbInner = fusedProduct;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingUbBroadcast::DoBroadcastTiling(
    const Ops::Base::BroadcastTilingParams& broadcastTilingParams,
    Ops::Base::BroadcastTilingData& broadcastTilingData)
{
    broadcastTilingData.shapeLen = dims.back().size();
    broadcastTilingData.dims = dims;
    broadcastTilingData.strides = strides;

    Ops::Base::BroadcastComputeParams computeParams;
    ge::graphStatus res = FindComputeParams(broadcastTilingParams, computeParams);
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    uint64_t ubSplitAxes = 0, curProduct = 0;
    res = CalcUbSplitParams(computeParams, ubSplitAxes, curProduct);
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    uint64_t maxElemNum = computeParams.bufferDivisor[0];
    uint32_t ubFormer = 0;
    uint64_t ubOuter = 0, ubTail = 0;
    res = CalcUbTailParams(ubSplitAxes, curProduct, maxElemNum, ubFormer, ubOuter, ubTail);
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    broadcastTilingData.ubSplitAxis = ubSplitAxes;
    broadcastTilingData.ubFormer = ubFormer;
    broadcastTilingData.ubOuter = ubOuter;
    broadcastTilingData.ubTail = ubTail;
    broadcastTilingData.elemNum = maxElemNum;

    ge::graphStatus blockRes = CalcBlockSplitParams(ubSplitAxes, ubOuter, broadcastTilingParams, broadcastTilingData);
    if (blockRes != ge::GRAPH_SUCCESS) {
        return blockRes;
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ClipByValue, ClipByValueTilingUbBroadcast, CLIP_BY_VALUE_UB_BROADCAST_TILING_PRIORITY);
} // namespace optiling