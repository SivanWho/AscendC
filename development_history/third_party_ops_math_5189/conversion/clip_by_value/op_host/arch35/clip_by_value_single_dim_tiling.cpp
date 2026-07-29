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
 * \file clip_by_value_single_dim_tiling.cpp
 * \brief
 */
#include <vector>
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "clip_by_value_tiling.h"
#include "clip_by_value_single_dim_tiling.h"
#include "op_host/math_tiling_templates_registry.h"
#include <string>

using namespace ge;
using namespace std;
namespace ClipByValueSigDim {
constexpr int64_t INOUT_PARAM_NUM = 4;
constexpr int64_t X_INDEX = 0;
constexpr int64_t MIN_INDEX = 1;
constexpr int64_t MAX_INDEX = 2;
constexpr int64_t Y_INDEX = 3;
constexpr int64_t OUTPUT_INDEX = 0;

constexpr uint64_t OPEN_DB_SIZE = 2;
constexpr uint64_t HALF_CORE_NUM_DIVIDE = 2;

constexpr int32_t ASS_ALIVE_NODE_NUM = 2;
constexpr int32_t SIG_DIM_ALIVE_NODE_NUM = 4;

constexpr int64_t PER_CORE_MIN_UB_BYTE = 8 * 1024;

constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
constexpr uint64_t CACHELINE_SIZE = 128;
constexpr int64_t LOW_UTILIZATION_DIM_MULTIPLIER = 2;
} // namespace ClipByValueSigDim

namespace optiling {

static constexpr uint64_t CLIP_BY_VALUE_SINGLE_DIM_TILING_KEY = 2000100;
static constexpr uint64_t CLIP_BY_VALUE_ASS_TILING_KEY = 2001100;

static constexpr uint64_t CLIP_BY_VALUE_SINGLE_DIM_TILING_PRIORITY = 1;

string ClipByValueTensorDesc2String(const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
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

string ClipByValueDebugTilingContext(gert::TilingContext* context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << ClipByValueTensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << ClipByValueTensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

bool ClipByValueTilingSingleDim::IsCapable()
{
    return isSigDim;
}

ge::graphStatus ClipByValueTilingSingleDim::GetPlatformInfo()
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

ge::graphStatus ClipByValueTilingSingleDim::CheckDtypesAndGetSize()
{
    auto xDesc = context_->GetInputDesc(ClipByValueSigDim::X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype = xDesc->GetDataType();

    auto minDesc = context_->GetInputDesc(ClipByValueSigDim::MIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minDesc);
    auto minDtype = minDesc->GetDataType();

    auto maxDesc = context_->GetInputDesc(ClipByValueSigDim::MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxDesc);
    auto maxDtype = maxDesc->GetDataType();

    auto yDesc = context_->GetOutputDesc(ClipByValueSigDim::OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
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

    if (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_INT32 &&
        xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16 &&
        xDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
            "float, float16, bfloat16, int32 and int64");
        return ge::GRAPH_FAILED;
    }
    dTypeSize = ge::GetSizeByDataType(xDtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingSingleDim::DoDimensionCollapse(
    std::vector<std::vector<int64_t>>& dims, std::vector<std::vector<int64_t>>& strides)
{
    auto xShape = context_->GetInputShape(ClipByValueSigDim::X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = Ops::Base::EnsureNotScalar(xShape->GetStorageShape());

    auto minShape = context_->GetInputShape(ClipByValueSigDim::MIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minShape);
    auto minStorageShape = Ops::Base::EnsureNotScalar(minShape->GetStorageShape());

    auto maxShape = context_->GetInputShape(ClipByValueSigDim::MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxShape);
    auto maxStorageShape = Ops::Base::EnsureNotScalar(maxShape->GetStorageShape());

    auto yShape = context_->GetOutputShape(ClipByValueSigDim::OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = Ops::Base::EnsureNotScalar(yShape->GetStorageShape());

    std::vector<gert::Shape> inShapes;
    inShapes.push_back(xStorageShape);
    inShapes.push_back(minStorageShape);
    inShapes.push_back(maxStorageShape);

    ge::graphStatus res = Ops::Base::DimensionCollapse(inShapes, yStorageShape, dims, strides);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "DimensionCollapse failed."), return res);

    if (dims.size() != ClipByValueSigDim::INOUT_PARAM_NUM) {
        OP_LOGE(context_->GetNodeName(),
            "DimensionCollapse failed. Check dims is illegal, out dims num %lu not equal to %ld, please check the "
            "shapes of x, clip_value_min, clip_value_max and y %s %s %s and %s",
            dims.size(), ClipByValueSigDim::INOUT_PARAM_NUM, Ops::Base::ToString(xStorageShape).c_str(),
            Ops::Base::ToString(minStorageShape).c_str(), Ops::Base::ToString(maxStorageShape).c_str(),
            Ops::Base::ToString(yStorageShape).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void ClipByValueTilingSingleDim::DetermineSigDimMode(const std::vector<std::vector<int64_t>>& dims)
{
    isSigDim = false;
    if (dims[ClipByValueSigDim::X_INDEX].size() == 1 &&
        dims[ClipByValueSigDim::MIN_INDEX].size() == 1 &&
        dims[ClipByValueSigDim::MAX_INDEX].size() == 1 &&
        dims[ClipByValueSigDim::Y_INDEX].size() == 1) {
        isSigDim = true;
        xDim = dims[ClipByValueSigDim::X_INDEX].front();
        minDim = dims[ClipByValueSigDim::MIN_INDEX].front();
        maxDim = dims[ClipByValueSigDim::MAX_INDEX].front();
        yDim = dims[ClipByValueSigDim::Y_INDEX].front();

        isAss = (minDim == 1 && maxDim == 1);
    }
}

ge::graphStatus ClipByValueTilingSingleDim::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "TilingContext: %s",
            optiling::ClipByValueDebugTilingContext(context_).c_str());

    ge::graphStatus res = CheckDtypesAndGetSize();
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    std::vector<std::vector<int64_t>> dims;
    std::vector<std::vector<int64_t>> strides;
    res = DoDimensionCollapse(dims, strides);
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    DetermineSigDimMode(dims);

    OP_LOGD(context_->GetNodeName(),
            "isSigDim: %d isAss: %d xDim: %ld minDim: %ld maxDim: %ld yDim: %ld",
            isSigDim, isAss, xDim, minDim, maxDim, yDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingSingleDim::CalcInitialUbParams(int64_t& ubFormer)
{
    int64_t dbUbSize = ubSize / ClipByValueSigDim::OPEN_DB_SIZE;
    int32_t aliveNum = isAss ? ClipByValueSigDim::ASS_ALIVE_NODE_NUM : ClipByValueSigDim::SIG_DIM_ALIVE_NODE_NUM;
    int64_t ubFormerByte = dbUbSize / aliveNum;
    int64_t ubFormerByteFloorAlign = (ubFormerByte / ClipByValueSigDim::CACHELINE_SIZE) * ClipByValueSigDim::CACHELINE_SIZE;
    ubFormer = ubFormerByteFloorAlign / dTypeSize;

    OP_CHECK_IF((ubFormer == 0),
                OP_LOGE(context_->GetNodeName(),
                        "ub size or cacheline check failed. ubSize: %lu cacheline: %lu dTypeSize: %u",
                        ubSize, ClipByValueSigDim::CACHELINE_SIZE, dTypeSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((coreNum == 0),
                OP_LOGE(context_->GetNodeName(), "core num check is 0, check failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((yDim == 0),
                OP_LOGE(context_->GetNodeName(), "tensor check is empty, check failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ClipByValueTilingSingleDim::CalcBlockParams(int64_t ubFormer, int64_t& ubOuter, int64_t& ubTail,
                                                  int64_t& blockFormer, int64_t& blockTail, int64_t& blockNum)
{
    ubOuter = (yDim + ubFormer - 1) / ubFormer;
    ubTail = yDim % ubFormer;
    ubTail = (ubTail == 0) ? ubFormer : ubTail;
    blockFormer = (ubOuter + coreNum - 1) / coreNum;
    blockTail = ubOuter % blockFormer;
    blockTail = (blockTail == 0) ? blockFormer : blockTail;
    blockNum = (ubOuter + blockFormer - 1) / blockFormer;
}

void ClipByValueTilingSingleDim::AdjustForLowUtilization(int64_t& ubFormer, int64_t& ubOuter, int64_t& ubTail,
                                                         int64_t& blockFormer, int64_t& blockTail, int64_t& blockNum)
{
    int64_t aliveNum = isAss ? ClipByValueSigDim::ASS_ALIVE_NODE_NUM : ClipByValueSigDim::SIG_DIM_ALIVE_NODE_NUM;
    if (static_cast<uint64_t>(blockNum) < (coreNum / ClipByValueSigDim::HALF_CORE_NUM_DIVIDE) &&
        (ubFormer * dTypeSize * aliveNum) > ClipByValueSigDim::PER_CORE_MIN_UB_BYTE) {
        int64_t dimPerCore = yDim * ClipByValueSigDim::LOW_UTILIZATION_DIM_MULTIPLIER / coreNum;
        int64_t alignDimPerCore = ((((dimPerCore * dTypeSize) + ClipByValueSigDim::CACHELINE_SIZE - 1) / ClipByValueSigDim::CACHELINE_SIZE) * ClipByValueSigDim::CACHELINE_SIZE) / dTypeSize;
        ubFormer = (ubFormer > alignDimPerCore) ? alignDimPerCore : ubFormer;

        int64_t lowestUbFormer =
            (((ClipByValueSigDim::PER_CORE_MIN_UB_BYTE / aliveNum) / ClipByValueSigDim::CACHELINE_SIZE) * ClipByValueSigDim::CACHELINE_SIZE) / dTypeSize;
        if (ubFormer < lowestUbFormer) {
            ubFormer = lowestUbFormer;
        }
        CalcBlockParams(ubFormer, ubOuter, ubTail, blockFormer, blockTail, blockNum);
    }
}

ge::graphStatus ClipByValueTilingSingleDim::DoOpTiling()
{
    int64_t ubFormer = 0;
    ge::graphStatus res = CalcInitialUbParams(ubFormer);
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    int64_t ubOuter = 0, ubTail = 0, blockFormer = 0, blockTail = 0, blockNum = 0;
    CalcBlockParams(ubFormer, ubOuter, ubTail, blockFormer, blockTail, blockNum);
    AdjustForLowUtilization(ubFormer, ubOuter, ubTail, blockFormer, blockTail, blockNum);

    tilingData.set_blockNum(blockNum);
    tilingData.set_ubFormer(ubFormer);
    tilingData.set_ubTail(ubTail);
    tilingData.set_blockFormer(blockFormer);
    tilingData.set_blockTail(blockTail);
    tilingData.set_xDim(xDim);
    tilingData.set_minDim(minDim);
    tilingData.set_maxDim(maxDim);
    tilingData.set_yDim(yDim);

    OP_LOGI(context_->GetNodeName(),
            "ClipByValue do tiling finish. coreNum: %ld ubSize: %lu cacheline: %lu "
            "blockNum: %ld ubFormer: %ld ubTail: %ld blockFormer: %ld blockTail: %ld "
            "xDim: %ld minDim: %ld maxDim: %ld yDim: %ld",
            coreNum, ubSize, ClipByValueSigDim::CACHELINE_SIZE, blockNum, ubFormer, ubTail, blockFormer, blockTail,
            xDim, minDim, maxDim, yDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingSingleDim::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ClipByValueTilingSingleDim::GetTilingKey() const
{
    if (isAss) {
        return CLIP_BY_VALUE_ASS_TILING_KEY;
    } else {
        return CLIP_BY_VALUE_SINGLE_DIM_TILING_KEY;
    }
}

ge::graphStatus ClipByValueTilingSingleDim::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = ClipByValueSigDim::MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTilingSingleDim::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData.get_blockNum());

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ClipByValue, ClipByValueTilingSingleDim, CLIP_BY_VALUE_SINGLE_DIM_TILING_PRIORITY);

} // namespace optiling