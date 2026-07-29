/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sort_tiling_arch35.cpp
 * \brief sort ac tiling impl
 */
#include "sort_tiling_arch35.h"

#include <iostream>
#include <limits>

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/sort_tiling_data.h"
#include "../../op_kernel/arch35/sort_tiling_key.h"

namespace optiling {
constexpr size_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
const uint32_t BIN_NUM = 256;                   // 直方图一次处理256B
const uint32_t SMALL_TILE_DATA_NUM = 1024;      // 测试数据得出一次至少处理1024，sort性能比较好
const uint32_t SIMT_UB = 32768;                 // 预留了32k给simt使用
const uint32_t TILE_DATA_NUM = 4096;            // merge_sort一次ub处理的数据量
const uint32_t SMALL_SORT_MAX_DATA_SIZE_FP32 = 4096; // fp32 走merge sort条件
const uint32_t SMALL_SORT_MAX_DATA_SIZE_FP16 = 1024; // fp16 或者 bf16 走merge sort条件
const uint32_t SMALL_AXIS_THRESHOLD = 512;
const uint32_t TWO_STAGE_RANK_INVERSE_MAX_N = 64;
const uint32_t MULTI_CORE_MERGE_SORT_MAX_SIZE = 32768;
const uint32_t SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT = 4095;  // DataCopy hardware limit for blockCount
const uint32_t MERGE_SORT_DEALING_LIST_NUM = 4;
const uint32_t MERGE_SORT_DATASIZE = 8;
const int64_t MERGE_SORT_WORKSPACE_PARAM = 5;
const int64_t ONE_CORE_DATA_SIZE = 2048;
const uint32_t SORT_STRUCT_SIZE_FP32 = 8;           // fp32 sort struct size (index + value)
const uint32_t MERGE_INTRA_CORE_SORT_ALIGN = 32;    // Sort/Extract API alignment requirement (elements)
// Beyond 4 merge rounds (>256 blocks), radix sort has little performance disadvantage.
const uint32_t MERGE_INTRA_CORE_MAX_BLOCKS = 256;
const uint32_t SORT32_SMALL_AXIS_THRESHOLD = 32;

struct SortTileInfo {
    uint32_t coreNumNeed = 0;
    uint32_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
    uint32_t ubSize = 0;
    uint32_t blockUbSize = 0;
    uint32_t dtypeSize = 0;
    uint32_t y2DtypeSize = 0;
    uint32_t maxCoreNum = 0;
    uint32_t numTileDataSize = 0;
    uint32_t sortLoopTimes = 0;
    uint32_t lastDimNeedCore = 0;
    uint32_t keyParams0 = 0;
    uint32_t keyParams1 = 0;
    uint32_t keyParams2 = 0;
    uint32_t keyParams3 = 0;
    uint32_t keyParams4 = 0;
    uint32_t keyParams5 = 0;
    uint32_t tmpUbSize = 0;
    bool isDescend = false;
    ge::DataType dataType = ge::DT_UINT8;
    uint32_t isInt32 = 0;
    int32_t xDimNum = 0;
    int64_t sortAxisNum = 1;
    int64_t unSortDimNum = 1;
};
static const std::map<ge::DataType, uint32_t> tilingDataTypeBitMap = {
    { ge::DT_INT64, 8 },  { ge::DT_INT32, 4 },   { ge::DT_INT16, 2 },  { ge::DT_INT8, 1 },
    { ge::DT_UINT64, 8 }, { ge::DT_UINT32, 4 },  { ge::DT_UINT16, 2 }, { ge::DT_UINT8, 1 },
    { ge::DT_FLOAT, 4 },  { ge::DT_FLOAT16, 2 }, { ge::DT_BF16, 2 }
};
static const std::map<ge::DataType, uint32_t> mergeType = { { ge::DT_FLOAT, 4 },
    { ge::DT_FLOAT16, 2 },
    { ge::DT_BF16, 2 } };

ge::graphStatus CheckInputAndOutput(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(SIMT_UB),
        OP_LOGE(context->GetNodeName(), "ubSize must be greater than %u, but is %lu", SIMT_UB, ubSize),
        return ge::GRAPH_FAILED);
    sortTileInfo.blockUbSize = Ops::Base::GetUbBlockSize(context);
    OP_LOGI(context->GetNodeName(), "ubSize is %ld, blockUbSize %u", ubSize, sortTileInfo.blockUbSize);
    sortTileInfo.ubSize = ubSize;
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape &inputShape = Ops::Base::EnsureNotScalar(inputShapePtr->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    const gert::Shape &outShape = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());
    auto yStorage1 = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage1);
    const gert::Shape &outShape1 = Ops::Base::EnsureNotScalar(yStorage1->GetStorageShape());
    OP_CHECK_IF(inputShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x, y1",
            "0", "The shape size of input x and output y1 should be positive"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShape != outShape1 || outShape != inputShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y1, y2",
            (Ops::Base::ToString(inputShape) + ", " + Ops::Base::ToString(outShape) + ", " + Ops::Base::ToString(outShape1)).c_str(),
            "The shape of input x, output y1 and y2 should be the same"),
        return ge::GRAPH_FAILED);
    int32_t xDimNum = inputShape.GetDimNum();
    sortTileInfo.xDimNum = xDimNum;
    int64_t sortAxisNum = inputShape.GetDim(xDimNum - 1);
    sortTileInfo.sortAxisNum = sortAxisNum;
    int64_t unSortDimNum = static_cast<int64_t>(1);
    for (uint32_t i = 0; i < static_cast<uint32_t>((xDimNum - 1)); i++) {
        int64_t dimSize = static_cast<int64_t>(inputShape.GetDim(i));
        unSortDimNum *= dimSize;
    }
    sortTileInfo.unSortDimNum = unSortDimNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortCheckParams(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    OP_CHECK_IF(CheckInputAndOutput(context, sortTileInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CheckInputAndOutput failed"), return ge::GRAPH_FAILED);
    auto inputDescPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
    ge::DataType dataType = inputDescPtr->GetDataType();
    OP_CHECK_IF(tilingDataTypeBitMap.count(dataType) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x",
            Ops::Base::ToString(dataType).c_str(),
            "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16, BF16"), return ge::GRAPH_FAILED);
    sortTileInfo.dataType = dataType;
    sortTileInfo.dtypeSize = tilingDataTypeBitMap.find(dataType)->second;
    auto outDescPtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr);
    auto y2DType = outDescPtr->GetDataType();
    auto outDescPtr0 = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr0);
    auto y1DType = outDescPtr0->GetDataType();
    OP_CHECK_IF((y2DType != ge::DT_INT64) && (y2DType != ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y2",
            Ops::Base::ToString(y2DType).c_str(), "INT32 or INT64"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(y1DType != dataType,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, y1",
            (Ops::Base::ToString(dataType) + ", " + Ops::Base::ToString(y1DType)).c_str(),
            "The dtype of input x should be the same as output y1"),
        return ge::GRAPH_FAILED);
    sortTileInfo.y2DtypeSize = tilingDataTypeBitMap.find(y2DType)->second;
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *isDescending = attrs->GetAttrPointer<bool>(1);
    const int64_t *sortAxisPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, isDescending);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortAxisPtr);
    int32_t sortAxis = static_cast<int32_t>(*sortAxisPtr);
    sortAxis = sortAxis < 0 ? (sortAxis + sortTileInfo.xDimNum) : sortAxis;
    OP_CHECK_IF(sortAxis != (sortTileInfo.xDimNum - 1),
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "axis",
            std::to_string(sortAxis).c_str(), "last dim (dimNum - 1)"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool TryCastInt64ToUint32(int64_t value, uint32_t &out)
{
    if (value < static_cast<int64_t>(0) ||
        value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

void SetSortTmpSize(ge::DataType dataType, uint32_t tileData, bool isDescend, SortTileInfo &sortTileInfo)
{
    int64_t realLen = std::min(sortTileInfo.sortAxisNum, static_cast<int64_t>(tileData));
    std::vector<int64_t> shapeVec = { realLen };
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTiling", "api of sort shape is %ld, maxUb is %u", realLen, maxValue);
    sortTileInfo.tmpUbSize = maxValue;
    return;
}

bool IsMergeSort(SortTileInfo &sortTileInfo)
{
    // fp16 和 bf16 进入条件和 fp32 进入条件不一样。
    // 基数排序对 fp16 和 bf16 只需要排序两轮，而对 fp32 需要排序四轮。
    // 因此对于 fp16 和 bf16 很较早进入基数排序和归并排序的算法临界点
    bool supportForFp16 = (sortTileInfo.sortAxisNum <= SMALL_SORT_MAX_DATA_SIZE_FP16) &&
                          (sortTileInfo.dataType == ge::DT_FLOAT16 || sortTileInfo.dataType == ge::DT_BF16);
    bool supportForFp32 =
        (sortTileInfo.sortAxisNum <= SMALL_SORT_MAX_DATA_SIZE_FP32) && (sortTileInfo.dataType == ge::DT_FLOAT);
    return supportForFp16 || supportForFp32;
}

bool IsSortMergeMultiCore(SortTileInfo &sortTileInfo)
{
    if (sortTileInfo.sortAxisNum > MULTI_CORE_MERGE_SORT_MAX_SIZE ||
        sortTileInfo.sortAxisNum <= SMALL_SORT_MAX_DATA_SIZE_FP32 || sortTileInfo.dataType != ge::DT_FLOAT) {
        return false;
    }
    uint32_t hCoreNum =
        static_cast<uint32_t>((sortTileInfo.sortAxisNum + ONE_CORE_DATA_SIZE - 1) / ONE_CORE_DATA_SIZE);
    bool isMuiltiCoreMergeSort =
        ((sortTileInfo.unSortDimNum > 0) && (hCoreNum > 0) &&
         (static_cast<uint64_t>(sortTileInfo.unSortDimNum) * hCoreNum <= sortTileInfo.maxCoreNum));
    return isMuiltiCoreMergeSort;
}

uint32_t ComputeMergeIntraCoreBlockSortSize(const SortTileInfo &sortTileInfo)
{
    // Phase 2 bottleneck: mergeIn(4 blocks) + mergeOut(4 blocks) of sort-struct data
    //   = MERGE_LIST_MAX_NUM * 2(in+out) * SORT_STRUCT_SIZE_FP32 * blockSortSize
    constexpr uint32_t MERGE_LIST_MAX_NUM = 4;
    constexpr uint32_t PHASE2_BYTES_PER_ELEM = MERGE_LIST_MAX_NUM * 2 * SORT_STRUCT_SIZE_FP32;  // = 64

    uint32_t blockSortSize = sortTileInfo.ubSize / PHASE2_BYTES_PER_ELEM;
    return (blockSortSize / MERGE_INTRA_CORE_SORT_ALIGN) * MERGE_INTRA_CORE_SORT_ALIGN;
}

uint32_t ComputeMergeIntraCoreExtractChunkSize(const SortTileInfo &sortTileInfo)
{
    // Phase 3: extractIn(x2) + outValue(x2) + outIdx(x2) + outIdxInt64(x2, int64 only)
    //   int32: (SORT_STRUCT_SIZE_FP32 + sizeof(float) + sizeof(int32_t)) * DOUBLE_BUFFER = 32
    //   int64: + sizeof(int64_t) * DOUBLE_BUFFER = 48
    constexpr uint32_t PHASE3_BYTES_PER_ELEM = (SORT_STRUCT_SIZE_FP32 + sizeof(float) +
        sizeof(int32_t) + sizeof(int64_t)) * 2;  // = 48, use int64 for safety

    uint32_t extractChunkSize = sortTileInfo.ubSize / PHASE3_BYTES_PER_ELEM;
    return (extractChunkSize / MERGE_INTRA_CORE_SORT_ALIGN) * MERGE_INTRA_CORE_SORT_ALIGN;
}

bool IsSortMergeIntraCoreAxisInRange(const SortTileInfo &sortTileInfo, uint32_t blockSortSize)
{
    if (blockSortSize == 0) {
        return false;
    }
    int64_t maxSortAxisNum = static_cast<int64_t>(blockSortSize) * MERGE_INTRA_CORE_MAX_BLOCKS;
    return sortTileInfo.sortAxisNum <= maxSortAxisNum;
}

/**
 * @brief Check if we should use intra-core block merge scenario
 * @details Conditions: fp32 + B >= maxCoreNum/2 + N > 4096 + blocksPerRow <= 256
 */
bool IsSortMergeIntraCore(SortTileInfo &sortTileInfo)
{
    if (sortTileInfo.dataType != ge::DT_FLOAT) {
        return false;
    }
    // for unSortDimNum which is too small, use radixmorecore kernel to maximum the usage of vector core
    if (sortTileInfo.unSortDimNum < sortTileInfo.maxCoreNum / 2) {
        return false;
    }
    if (sortTileInfo.sortAxisNum <= SMALL_SORT_MAX_DATA_SIZE_FP32) {
        return false;
    }

    uint32_t blockSortSize = ComputeMergeIntraCoreBlockSortSize(sortTileInfo);
    if (!IsSortMergeIntraCoreAxisInRange(sortTileInfo, blockSortSize)) {
        return false;
    }

    if (ComputeMergeIntraCoreExtractChunkSize(sortTileInfo) == 0) {
        return false;
    }
    uint64_t maxBatch = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) *
        static_cast<uint64_t>(sortTileInfo.maxCoreNum);
    if (sortTileInfo.unSortDimNum <= 0 || static_cast<uint64_t>(sortTileInfo.unSortDimNum) > maxBatch) {
        return false;
    }

    return true;
}

// Align bytes up to alignBytes and return false if the aligned result cannot fit in uint32_t.
static bool TryAlignBytesToUint32(uint64_t bytes, uint32_t alignBytes, uint32_t &alignedBytes)
{
    if (bytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    uint64_t alignedBytes64 = Ops::Base::CeilAlign<uint64_t>(bytes, static_cast<uint64_t>(alignBytes));
    if (alignedBytes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    alignedBytes = static_cast<uint32_t>(alignedBytes64);
    return true;
}

bool IsRadixSortOneCore(SortTileInfo &sortTileInfo)
{
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        return false;
    }
    // One-core radix sort keeps one input value queue, one sorted value queue, and one output index queue in UB.
    uint32_t xUbSize = 0;
    uint64_t xBytes = static_cast<uint64_t>(sortTileInfo.sortAxisNum) *
                      static_cast<uint64_t>(sortTileInfo.dtypeSize);
    if (!TryAlignBytesToUint32(xBytes, sortTileInfo.blockUbSize, xUbSize)) {
        return false;
    }

    uint32_t y2UbSize = 0;
    uint64_t y2Bytes = static_cast<uint64_t>(sortTileInfo.sortAxisNum) * sizeof(int32_t);
    if (!TryAlignBytesToUint32(y2Bytes, sortTileInfo.blockUbSize, y2UbSize)) {
        return false;
    }

    // Sort API writes uint32 indices first. For int64 output, reserve the other half for Cast result.
    uint32_t halfNum = y2UbSize / static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.y2DtypeSize == static_cast<uint32_t>(sizeof(int64_t))) {
        y2UbSize = y2UbSize * static_cast<uint32_t>(sizeof(int64_t) / sizeof(int32_t));
    }
    sortTileInfo.keyParams0 = xUbSize;
    sortTileInfo.keyParams1 = y2UbSize;
    sortTileInfo.keyParams2 = halfNum;
    sortTileInfo.keyParams3 = 1;
    // keyParams3 records whether the queues can use double buffer after reserving Sort tmp UB.
    int64_t oneBufferQueSize = static_cast<int64_t>(xUbSize) * 2 + static_cast<int64_t>(y2UbSize);
    int64_t remainUb = static_cast<int64_t>(sortTileInfo.ubSize) - oneBufferQueSize;
    if (remainUb <= static_cast<int64_t>(0)) {
        return false;
    }
    remainUb = (remainUb / static_cast<int64_t>(sortTileInfo.blockUbSize)) *
        static_cast<int64_t>(sortTileInfo.blockUbSize);
    SetSortTmpSize(sortTileInfo.dataType, static_cast<uint32_t>(sortTileInfo.sortAxisNum),
        sortTileInfo.isDescend, sortTileInfo);
    int64_t tmpUb = static_cast<int64_t>(sortTileInfo.tmpUbSize);
    OP_LOGI("RadixSortTiling", "remainUb is %ld, tmpUb is %ld", remainUb, tmpUb);
    if (tmpUb > remainUb) {
        return false;
    }

    int64_t doubleBufferRemainUb = static_cast<int64_t>(sortTileInfo.ubSize) - oneBufferQueSize * 2;
    doubleBufferRemainUb = (doubleBufferRemainUb / static_cast<int64_t>(sortTileInfo.blockUbSize)) *
        static_cast<int64_t>(sortTileInfo.blockUbSize);
    if (tmpUb <= doubleBufferRemainUb) {
        sortTileInfo.keyParams3 = 2;
    }
    OP_LOGI("RadixSortTiling", "radix one-core bufferNum is %u", sortTileInfo.keyParams3);
    return true;
}

bool IsAxisOneCopy(const SortTileInfo &sortTileInfo)
{
    return sortTileInfo.sortAxisNum == static_cast<int64_t>(1);
}

ge::graphStatus GetAxisOneCopy(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    // double buffer
    uint64_t bytesPerElem = static_cast<uint64_t>(2) *
        (static_cast<uint64_t>(sortTileInfo.dtypeSize) + static_cast<uint64_t>(sortTileInfo.y2DtypeSize));
    if (bytesPerElem == 0) {
        OP_LOGE(context->GetNodeName(), "bytesPerElem is 0, invalid dtype configuration");
        return ge::GRAPH_FAILED;
    }
    uint64_t copyElemsPerLoop64 = static_cast<uint64_t>(sortTileInfo.ubSize) / bytesPerElem;
    if (copyElemsPerLoop64 == 0) {
        OP_LOGE(context->GetNodeName(), "copyElemsPerLoop is 0, ub is too small for axis-one copy");
        return ge::GRAPH_FAILED;
    }
    if (copyElemsPerLoop64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "copyElemsPerLoop exceeds uint32_t limit");
        return ge::GRAPH_FAILED;
    }
    uint32_t copyElemsPerLoop = static_cast<uint32_t>(copyElemsPerLoop64);
    uint64_t totalElems = static_cast<uint64_t>(sortTileInfo.unSortDimNum) *
        static_cast<uint64_t>(sortTileInfo.sortAxisNum);
    uint64_t loopTimes64 = (totalElems + copyElemsPerLoop64 - 1) / copyElemsPerLoop64;
    if (loopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "loopTimes exceeds uint32_t limit");
        return ge::GRAPH_FAILED;
    }
    uint32_t loopTimes = static_cast<uint32_t>(loopTimes64);

    uint32_t coreNumNeed = std::min(sortTileInfo.maxCoreNum, loopTimes);
    sortTileInfo.numTileDataSize = copyElemsPerLoop;
    sortTileInfo.keyParams0 = copyElemsPerLoop;
    sortTileInfo.keyParams1 = loopTimes;
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.unsortedDimParallel = coreNumNeed;
    sortTileInfo.lastDimTileNum = 1;
    sortTileInfo.lastDimNeedCore = 1;
    sortTileInfo.sortLoopTimes = Ops::Base::CeilDiv(static_cast<int64_t>(loopTimes), static_cast<int64_t>(coreNumNeed));
    sortTileInfo.tmpUbSize = 0;

    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    OP_LOGI("AxisOneCopyTiling", "totalElems %lu, copyElemsPerLoop %u, loopTimes %u, coreNumNeed %u",
        totalElems, sortTileInfo.keyParams0, sortTileInfo.keyParams1, coreNumNeed);
    return ge::GRAPH_SUCCESS;
}

// Returns align_up(elemCount * elemBytes, blockUbSize), or 0 on overflow/zero block
static uint32_t AlignRowBytes(uint32_t elemCount, uint32_t elemBytes, uint32_t blockUbSize)
{
    uint64_t raw = static_cast<uint64_t>(elemCount) * static_cast<uint64_t>(elemBytes);
    if (raw > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return 0;
    }
    uint64_t aligned = Ops::Base::CeilAlign<uint64_t>(raw, static_cast<uint64_t>(blockUbSize));
    if (aligned > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return 0;
    }
    return static_cast<uint32_t>(aligned);
}

uint32_t ComputeInsertionBytesPerSeg(const SortTileInfo &sortTileInfo)
{
    uint32_t axisLen = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    uint32_t alignUnit = sortTileInfo.blockUbSize;
    uint32_t valueBytes = AlignRowBytes(axisLen, sortTileInfo.dtypeSize, alignUnit);
    uint32_t idxBytes = AlignRowBytes(axisLen, sortTileInfo.y2DtypeSize, alignUnit);
    if (valueBytes == 0 || idxBytes == 0) {
        return 0;
    }
    uint64_t bytesPerSeg = static_cast<uint64_t>(valueBytes) + static_cast<uint64_t>(idxBytes);
    if (sortTileInfo.dataType == ge::DT_BF16) {
        uint32_t castBytes = AlignRowBytes(axisLen, static_cast<uint32_t>(sizeof(int16_t)), alignUnit);
        if (castBytes == 0) {
            return 0;
        }
        uint32_t castRowElems = castBytes / static_cast<uint32_t>(sizeof(int16_t));
        uint64_t bf16ValueBytes = static_cast<uint64_t>(castRowElems) * static_cast<uint64_t>(sizeof(float));
        if (bf16ValueBytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return 0;
        }
        valueBytes = static_cast<uint32_t>(bf16ValueBytes);
        bytesPerSeg = static_cast<uint64_t>(valueBytes) + static_cast<uint64_t>(idxBytes);
        bytesPerSeg += static_cast<uint64_t>(castBytes);
    }
    if (bytesPerSeg > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return 0;
    }
    return static_cast<uint32_t>(bytesPerSeg);
}

bool EstimateInsertionBatching(const SortTileInfo &sortTileInfo, uint32_t fullCoreSegs,
    uint32_t &batchSize, uint32_t &batchNum, uint32_t &blockDim, uint32_t &sUbMax)
{
    uint32_t usableUb = sortTileInfo.ubSize - SIMT_UB;
    uint32_t bytesPerSeg = ComputeInsertionBytesPerSeg(sortTileInfo);
    if (bytesPerSeg == 0 || usableUb < bytesPerSeg) {
        return false;
    }
    sUbMax = usableUb / bytesPerSeg;
    if (sUbMax == 0) {
        return false;
    }

    // INSERTION kernel supports in-core loop for multiple batches; batchSize limited by UB and DataCopy blockCount.
    batchSize = std::min({fullCoreSegs, sUbMax, SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT});
    uint32_t batchNumCandidate = 0;
    if (!TryCastInt64ToUint32(Ops::Base::CeilDiv(sortTileInfo.unSortDimNum, static_cast<int64_t>(batchSize)),
        batchNumCandidate)) {
        return false;
    }
    batchNum = batchNumCandidate;
    blockDim = std::min(sortTileInfo.maxCoreNum, batchNum);
    if (batchNum == 0) {
        return false;
    }
    return true;
}

enum class SmallAxisRouteKind : uint8_t {
    NONE = 0,
    INSERTION,
    TWO_STAGE
};

struct SmallAxisRoutePlan {
    SmallAxisRouteKind kind = SmallAxisRouteKind::NONE;
    uint32_t batchSize = 0;
    uint32_t batchNum = 0;
    uint32_t blockDim = 0;
    uint32_t tmpUbSize = 0;
    bool useRankInverse = false;
};

// Per-dtype small-axis routing table.
struct SmallAxisRule {
    ge::DataType dtype;
    uint32_t insertionMaxN;  // max sortAxisNum for INSERTION; 0 = unsupported
    uint32_t twoStageMaxN;   // max sortAxisNum for TWO_STAGE; 0 = unsupported
    // INSERTION concurrency tiers: {maxN, minSegs} pairs, terminated by {0, 0}.
    // A hit requires: N <= maxN && fullCoreSegs >= minSegs for some tier.
    uint32_t insertionTiers[4][2];
    // TWO_STAGE concurrency tiers: {maxN, minSegs} pairs, ascending maxN, terminated by {0, 0}.
    // Same lookup as INSERTION: find first tier where N <= maxN, then check fullCoreSegs >= minSegs.
    uint32_t twoStageTiers[4][2];
};

constexpr SmallAxisRule kSmallAxisRules[] = {
    // Small-axis thresholds are empirical performance breakpoints.
    // INSERTION is preferred only for very small N and enough full-core segments; otherwise its serial compare cost
    // grows quickly. TWO_STAGE covers wider N ranges, but each tier still requires enough segments to amortize
    // Sort API and SIMT scatter overhead. Each {maxN, minSegs} pair means this route is used only when
    // sortAxisNum <= maxN and each active core can receive at least minSegs segments.
    // dtype, insertionMaxN, twoStageMaxN, insertion tiers, two-stage tiers
    { ge::DT_INT64, 16, 512,
        { { 8, 1 }, { 16, 4 }, { 0, 0 } },
        { { 15, 8 }, { 128, 4 }, { 512, 8 }, { 0, 0 } } },
    { ge::DT_UINT64, 16, 512,
        { { 8, 1 }, { 16, 4 }, { 0, 0 } },
        { { 15, 8 }, { 128, 4 }, { 512, 8 }, { 0, 0 } } },
    { ge::DT_INT32, 11, 384,
        { { 8, 2 }, { 11, 4 }, { 0, 0 } },
        { { 11, 8 }, { 64, 4 }, { 384, 12 }, { 0, 0 } } },
    { ge::DT_UINT32, 11, 384,
        { { 8, 2 }, { 11, 4 }, { 0, 0 } },
        { { 11, 8 }, { 64, 4 }, { 384, 12 }, { 0, 0 } } },
    { ge::DT_INT16, 8, 192,
        { { 4, 2 }, { 8, 4 }, { 0, 0 } },
        { { 7, 8 }, { 64, 4 }, { 192, 12 }, { 0, 0 } } },
    { ge::DT_UINT16, 8, 192,
        { { 4, 2 }, { 8, 4 }, { 0, 0 } },
        { { 7, 8 }, { 64, 4 }, { 192, 12 }, { 0, 0 } } },
    { ge::DT_INT8, 8, 112,
        { { 4, 2 }, { 8, 7 }, { 0, 0 } },
        { { 3, 8 }, { 64, 7 }, { 112, 16 }, { 0, 0 } } },
    { ge::DT_UINT8, 8, 112,
        { { 4, 2 }, { 8, 7 }, { 0, 0 } },
        { { 3, 8 }, { 64, 7 }, { 112, 16 }, { 0, 0 } } },
    { ge::DT_FLOAT, 8, 24,
        { { 4, 16 }, { 8, 48 }, { 0, 0 } },
        { { 24, 64 }, { 0, 0 } } },
    { ge::DT_BF16, 8, 54,
        { { 4, 16 }, { 8, 48 }, { 0, 0 } },
        { { 54, 64 }, { 0, 0 } } },
    { ge::DT_FLOAT16, 8, 54,
        { { 4, 16 }, { 8, 48 }, { 0, 0 } },
        { { 54, 64 }, { 0, 0 } } },
};
constexpr size_t kNumSmallAxisRules = sizeof(kSmallAxisRules) / sizeof(kSmallAxisRules[0]);

// Look up minSegs from ascending {maxN, minSegs} tiers. Returns UINT32_MAX if no match.
static uint32_t LookupMinSegs(const uint32_t (*tiers)[2], uint32_t axisNum)
{
    constexpr uint32_t kMaxTiers = 4;
    for (uint32_t i = 0; i < kMaxTiers && tiers[i][0] != 0; ++i) {
        if (axisNum <= tiers[i][0]) return tiers[i][1];
    }
    return UINT32_MAX;
}

// Check whether INSERTION is feasible for the given (rule, N, fullCoreSegs).
// Returns true and fills batch params on success.
bool InsertionFeasible(const SortTileInfo &sortTileInfo, const SmallAxisRule &rule,
                       uint32_t axisNum, uint32_t fullCoreSegs,
                       uint32_t &batchSize, uint32_t &batchNum, uint32_t &blockDim)
{
    if (axisNum > rule.insertionMaxN) {
        return false;
    }

    // INSERTION defers to TWO_STAGE when TWO_STAGE is viable.
    if (rule.twoStageMaxN > 0 && fullCoreSegs >= LookupMinSegs(rule.twoStageTiers, axisNum)) {
        return false;
    }

    // Check concurrency tiers
    if (fullCoreSegs < LookupMinSegs(rule.insertionTiers, axisNum)) {
        return false;
    }

    // UB budget check
    uint32_t ubCap;
    if (!EstimateInsertionBatching(sortTileInfo, fullCoreSegs, batchSize, batchNum, blockDim, ubCap)) {
        return false;
    }
    return true;
}

void FillSmallAxisBatched(gert::TilingContext *context, SortTileInfo &sortTileInfo,
    const SmallAxisRoutePlan &plan)
{
    sortTileInfo.ubSize = sortTileInfo.ubSize - SIMT_UB;
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    sortTileInfo.keyParams0 = plan.batchSize;
    sortTileInfo.keyParams1 = plan.batchNum;
    sortTileInfo.keyParams2 = plan.useRankInverse ? 1U : 0U;
    sortTileInfo.coreNumNeed = plan.blockDim;
    sortTileInfo.tmpUbSize = plan.tmpUbSize;
    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
}

bool UseTwoStageRankInverse(const SortTileInfo &sortTileInfo)
{
    return sortTileInfo.sortAxisNum <= static_cast<int64_t>(TWO_STAGE_RANK_INVERSE_MAX_N);
}

uint64_t EstimateTwoStageUbBytes(const SortTileInfo &sortTileInfo, uint32_t totalElems, uint32_t sortTmpUb)
{
    uint32_t alignUnit = sortTileInfo.blockUbSize;
    uint32_t dtypeBytes = sortTileInfo.dtypeSize;
    // Value UB layout follows SortSmallAxisTwoStage::Init:
    // inputBuf_ + stage1ValueBuf_. finalValues_ aliases inputBuf_, so it does not need another value buffer.
    uint64_t alignedValueBytes = Ops::Base::CeilAlign<uint64_t>(
        static_cast<uint64_t>(totalElems) * dtypeBytes, static_cast<uint64_t>(alignUnit));

    // uint32 index/order buffers: stage1OrderBuf_ is always needed. Rank-inverse mode adds rankInverseBuf_;
    // two-stage-sort mode adds rankInverseBuf_ as stage2KeysOut_ plus stage2OrderBuf_.
    uint64_t alignedIdxBytes = Ops::Base::CeilAlign<uint64_t>(
        static_cast<uint64_t>(totalElems) * sizeof(uint32_t), static_cast<uint64_t>(alignUnit));

    // finalIdxBuf_ stores OutIdxT final indices, and the non-rank-inverse path reuses it as uint32 stage2KeysIn_.
    // Allocate by the larger element size so both views fit: int32/uint32 use 4 bytes, int64 uses 8 bytes.
    uint32_t aliasBytes = std::max(static_cast<uint32_t>(sizeof(uint32_t)), sortTileInfo.y2DtypeSize);
    uint64_t alignedAliasBytes = Ops::Base::CeilAlign<uint64_t>(
        static_cast<uint64_t>(totalElems) * aliasBytes, static_cast<uint64_t>(alignUnit));

    uint64_t idxBufferCount = UseTwoStageRankInverse(sortTileInfo) ? 2U : 3U;
    return alignedValueBytes * 2 + alignedIdxBytes * idxBufferCount + alignedAliasBytes +
        static_cast<uint64_t>(sortTmpUb);
}

static bool TryTwoStageBatchCandidate(const SortTileInfo &sortTileInfo, uint32_t candidate,
    uint32_t &tempBatchNum, uint32_t &tempBlockDim, uint32_t &sortTmpUb)
{
    uint64_t totalElems = static_cast<uint64_t>(candidate) * static_cast<uint64_t>(sortTileInfo.sortAxisNum);
    if (totalElems > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    // Stage-2 key is encoded as segId * totalElems + rank. Example: batchSize=4, N=16 gives
    // totalElems=64, and the last segment may use keys near 3 * 64 + rank; keep the key range in uint32.
    uint64_t stage2KeyMax = static_cast<uint64_t>(candidate) * totalElems;
    if (stage2KeyMax > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }

    uint32_t castTotalElems = static_cast<uint32_t>(totalElems);
    SortTileInfo tmpTileInfo = sortTileInfo;
    tmpTileInfo.sortAxisNum = static_cast<int64_t>(castTotalElems);
    SetSortTmpSize(sortTileInfo.dataType, castTotalElems, false, tmpTileInfo);
    sortTmpUb = tmpTileInfo.tmpUbSize;
    // Without rank-inverse, stage 2 sorts uint32 keys, so reserve tmp UB for the larger of the two Sort calls.
    if (!UseTwoStageRankInverse(sortTileInfo)) {
        uint32_t stage1SortTmpUb = sortTmpUb;
        SetSortTmpSize(ge::DT_UINT32, castTotalElems, false, tmpTileInfo);
        sortTmpUb = std::max(stage1SortTmpUb, tmpTileInfo.tmpUbSize);
    }
    uint64_t totalBytes = EstimateTwoStageUbBytes(sortTileInfo, castTotalElems, sortTmpUb);
    if (totalBytes + static_cast<uint64_t>(SIMT_UB) > static_cast<uint64_t>(sortTileInfo.ubSize)) {
        return false;
    }

    if (!TryCastInt64ToUint32(Ops::Base::CeilDiv(sortTileInfo.unSortDimNum, static_cast<int64_t>(candidate)),
        tempBatchNum)) {
        return false;
    }
    tempBlockDim = std::min(sortTileInfo.maxCoreNum, tempBatchNum);
    return true;
}

static uint32_t EstimateTwoStageMaxBatchByUb(const SortTileInfo &sortTileInfo)
{
    uint32_t aliasBytes = std::max(static_cast<uint32_t>(sizeof(uint32_t)), sortTileInfo.y2DtypeSize);
    uint64_t idxBufferCount = UseTwoStageRankInverse(sortTileInfo) ? 2U : 3U;
    uint64_t minBytesPerElem =
        static_cast<uint64_t>(sortTileInfo.dtypeSize) * 2U +
        static_cast<uint64_t>(sizeof(uint32_t)) * idxBufferCount + static_cast<uint64_t>(aliasBytes);
    uint64_t usableUb = static_cast<uint64_t>(sortTileInfo.ubSize - SIMT_UB);
    uint64_t maxElemsByUb = usableUb / minBytesPerElem;
    uint64_t maxBatchByUb = maxElemsByUb / static_cast<uint64_t>(sortTileInfo.sortAxisNum);
    uint64_t maxBatch =
        std::min<uint64_t>(static_cast<uint64_t>(sortTileInfo.unSortDimNum), maxBatchByUb);
    return static_cast<uint32_t>(std::min<uint64_t>(
        maxBatch, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
}

// Estimate TWO_STAGE batching params: batchSize, batchNum, blockDim, tmpUbSize.
// Strategy: iterate downward from UB-limit to find max feasible batch.
// Alignment optimization: if batchNum%blockDim remainder < 60% of blockDim,
// search for aligned plan (batchNum%blockDim==0) to avoid load imbalance.
static bool EstimateTwoStageBatching(const SortTileInfo &sortTileInfo, uint32_t &batchSize,
    uint32_t &batchNum, uint32_t &blockDim, uint32_t &tmpUbSize)
{
    int64_t fullCoreSegs64 = Ops::Base::CeilDiv(sortTileInfo.unSortDimNum, static_cast<int64_t>(sortTileInfo.maxCoreNum));
    uint32_t fullCoreSegs = static_cast<uint32_t>(std::min<int64_t>(fullCoreSegs64, static_cast<int64_t>(std::numeric_limits<uint32_t>::max())));
    // UB caps the flattened batch size; fullCoreSegs caps it so all active cores have work.
    uint32_t maxBatchCandidate = std::min(EstimateTwoStageMaxBatchByUb(sortTileInfo), fullCoreSegs);
    if (maxBatchCandidate == 0) {
        return false;
    }

    struct BatchPlan {
        uint32_t size = 0;
        uint32_t num = 0;
        uint32_t block = 0;
        uint32_t tmpUb = 0;
    };
    BatchPlan maxPlan;
    BatchPlan alignedPlan;
    bool needAligned = false;
    bool hasMaxPlan = false;
    bool hasAlignedPlan = false;

    // First feasible candidate is the largest UB-safe batch; keep searching only if tail load is too small.
    for (uint32_t candidate = maxBatchCandidate; candidate >= 1U; --candidate) {
        uint32_t tempBatchNum = 0;
        uint32_t tempBlockDim = 0;
        uint32_t sortTmpUb = 0;
        if (!TryTwoStageBatchCandidate(sortTileInfo, candidate, tempBatchNum, tempBlockDim, sortTmpUb)) {
            continue;
        }

        if (!hasMaxPlan) {
            maxPlan = {candidate, tempBatchNum, tempBlockDim, sortTmpUb};
            hasMaxPlan = true;
            uint32_t remainder = maxPlan.num % maxPlan.block;
            // If the last wave uses fewer than 60% cores, try a smaller batch that divides work evenly.
            needAligned = (remainder != 0) && (remainder * 100 < maxPlan.block * 60);
            if (!needAligned) {
                break;
            }
        }

        if (needAligned && tempBatchNum % tempBlockDim == 0) {
            alignedPlan = {candidate, tempBatchNum, tempBlockDim, sortTmpUb};
            hasAlignedPlan = true;
            break;
        }
    }

    if (!hasMaxPlan) {
        return false;
    }
    // Prefer the aligned plan only when it exists; otherwise keep the maximum feasible batch.
    auto &result = (needAligned && hasAlignedPlan) ? alignedPlan : maxPlan;
    batchSize = result.size;
    batchNum = result.num;
    blockDim = result.block;
    tmpUbSize = result.tmpUb;
    return batchNum > 0 && blockDim > 0;
}

// Check whether TWO_STAGE is feasible for the given (rule, N, fullCoreSegs).
// Returns true and fills batch params on success.
bool TwoStageFeasible(const SortTileInfo &sortTileInfo, const SmallAxisRule &rule,
                      uint32_t axisNum, uint32_t fullCoreSegs,
                      uint32_t &batchSize, uint32_t &batchNum, uint32_t &blockDim, uint32_t &tmpUbSize)
{
    if (rule.twoStageMaxN == 0 || axisNum > rule.twoStageMaxN) {
        return false;
    }
    if (axisNum > SMALL_AXIS_THRESHOLD) {
        return false;
    }
    if (fullCoreSegs < LookupMinSegs(rule.twoStageTiers, axisNum)) {
        return false;
    }

    return EstimateTwoStageBatching(sortTileInfo, batchSize, batchNum, blockDim, tmpUbSize);
}

bool SelectSmallAxisRoute(const SortTileInfo &sortTileInfo, SmallAxisRoutePlan &routePlan)
{
    if (sortTileInfo.sortAxisNum <= 1) {
        return false;
    }

    uint32_t sortAxisSize = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    int64_t fullCoreSegs64 =
        Ops::Base::CeilDiv(sortTileInfo.unSortDimNum, static_cast<int64_t>(sortTileInfo.maxCoreNum));
    uint32_t fullCoreSegs = static_cast<uint32_t>(std::min<int64_t>(
        fullCoreSegs64, static_cast<int64_t>(std::numeric_limits<uint32_t>::max())));
    const SmallAxisRule *rule = nullptr;
    for (size_t i = 0; i < kNumSmallAxisRules; ++i) {
        if (kSmallAxisRules[i].dtype == sortTileInfo.dataType) {
            rule = &kSmallAxisRules[i];
            break;
        }
    }
    if (rule == nullptr) {
        return false;
    }

    uint32_t insertionBatchSize = 0;
    uint32_t insertionBatchNum = 0;
    uint32_t insertionBlockDim = 0;
    uint32_t twoStageBatchSize = 0;
    uint32_t twoStageBatchNum = 0;
    uint32_t twoStageBlockDim = 0;
    uint32_t twoStageTmpUbSize = 0;

    bool insertionFeasible = InsertionFeasible(sortTileInfo, *rule, sortAxisSize, fullCoreSegs,
        insertionBatchSize, insertionBatchNum, insertionBlockDim);
    bool twoStageFeasible = TwoStageFeasible(sortTileInfo, *rule, sortAxisSize, fullCoreSegs,
        twoStageBatchSize, twoStageBatchNum, twoStageBlockDim, twoStageTmpUbSize);

    OP_LOGI("SmallAxisRoute",
        "dtype %d axis %u, fullCoreSegs=%u, "
        "insertion(batch=%u,num=%u,block=%u,ok=%d) "
        "twoStage(batch=%u,num=%u,block=%u,ok=%d)",
        static_cast<int32_t>(sortTileInfo.dataType), sortAxisSize, fullCoreSegs,
        insertionBatchSize, insertionBatchNum, insertionBlockDim, static_cast<int32_t>(insertionFeasible),
        twoStageBatchSize, twoStageBatchNum, twoStageBlockDim, static_cast<int32_t>(twoStageFeasible));

    // TWO_STAGE takes over high-parallel small-axis cases.
    if (twoStageFeasible) {
        routePlan = {SmallAxisRouteKind::TWO_STAGE, twoStageBatchSize, twoStageBatchNum, twoStageBlockDim,
            twoStageTmpUbSize, UseTwoStageRankInverse(sortTileInfo)};
        return true;
    }

    if (insertionFeasible) {
        routePlan = {SmallAxisRouteKind::INSERTION, insertionBatchSize, insertionBatchNum, insertionBlockDim};
        return true;
    }

    return false;
}

uint32_t ComputeRemainUb(SortTileInfo &sortTileInfo, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint32_t tmpUb = sortTileInfo.ubSize - (ubExtra + tileFactor * tileData);
    return tmpUb;
}

void AdjTmpUb(SortTileInfo &sortTileInfo, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint32_t remainUbNew = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor) - sortTileInfo.tmpUbSize;
    remainUbNew = remainUbNew > sortTileInfo.blockUbSize ? (remainUbNew - sortTileInfo.blockUbSize) : uint32_t(0);
    uint32_t alignUbSize = (remainUbNew / sortTileInfo.blockUbSize) * sortTileInfo.blockUbSize;
    OP_LOGI("RadixSortTiling", "alignUbSize %u", alignUbSize);
    sortTileInfo.tmpUbSize = sortTileInfo.tmpUbSize + alignUbSize; // 剩余的ub都给tmpUbsize
}

void ComputeTileDataOne(SortTileInfo &sortTileInfo, uint32_t lastDimTileNum,  uint32_t ubExtra, uint32_t &tileData,
                        uint32_t tileFactor)
{
    uint32_t allCore = Ops::Base::CeilAlign<uint32_t>(lastDimTileNum, sortTileInfo.maxCoreNum);
    uint32_t newTileData = Ops::Base::CeilDiv(sortTileInfo.sortAxisNum, int64_t(allCore));
    tileData = Ops::Base::CeilAlign<uint32_t>(newTileData, BIN_NUM);
    tileData = std::max(tileData, SMALL_TILE_DATA_NUM);
    SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
    AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
    return;
}

bool NeedAdjTileData(SortTileInfo &sortTileInfo, uint32_t &tileData, uint32_t lastDimTileNum, uint32_t ubExtra,
                     uint32_t tileFactor)
{
    if (sortTileInfo.unSortDimNum == int64_t(1) && lastDimTileNum == uint32_t(1)) {
        OP_LOGI("RadixSortTiling", "unSortDimNum and lastDimTileNum is 1");
        uint32_t newTileData = Ops::Base::CeilDiv(sortTileInfo.sortAxisNum, int64_t(sortTileInfo.maxCoreNum));
        newTileData = Ops::Base::CeilAlign<uint32_t>(newTileData, BIN_NUM);
        tileData = std::max(newTileData, SMALL_TILE_DATA_NUM);
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return true;
    }
    if (sortTileInfo.unSortDimNum == int64_t(1) || (lastDimTileNum >= sortTileInfo.maxCoreNum)) {
        // b为1时，尽量均匀分核，同时保证处理的最小的tile_data为1024
        OP_LOGI("RadixSortTiling", "unSortDimNum is 1 and lastDimTileNum greater than allCore");
        ComputeTileDataOne(sortTileInfo, lastDimTileNum, ubExtra, tileData, tileFactor);
        return true;
    }
    if (sortTileInfo.unSortDimNum > int64_t(1) && sortTileInfo.unSortDimNum < int64_t(sortTileInfo.maxCoreNum) &&
        lastDimTileNum == uint32_t(1)) {
        OP_LOGI("RadixSortTiling", "unSortDimNum greater than 1,and unSortDimNum small and lastDimTileNum is one");
        uint32_t hCore = sortTileInfo.maxCoreNum / static_cast<uint32_t>(sortTileInfo.unSortDimNum);
        uint32_t hTileData = static_cast<uint32_t>(sortTileInfo.sortAxisNum) / hCore;
        tileData = Ops::Base::CeilAlign<uint32_t>(hTileData, BIN_NUM);
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return true;
    }
    if (sortTileInfo.unSortDimNum > int64_t(1) && lastDimTileNum > uint32_t(1)) {
        // b大于1且h轴循环次数小于总核数，也就是b轴核数大于1
        OP_LOGI("RadixSortTiling", "unSortDimNum is one, lastDimTileNum greater than one");
        int64_t newTileData = sortTileInfo.sortAxisNum / int64_t(lastDimTileNum);
        tileData = Ops::Base::CeilAlign<uint32_t>(static_cast<uint32_t>(newTileData), BIN_NUM);
        lastDimTileNum = Ops::Base::CeilDiv(sortTileInfo.sortAxisNum, int64_t(tileData));
        uint32_t bCore = lastDimTileNum == 0 ? sortTileInfo.maxCoreNum : sortTileInfo.maxCoreNum / lastDimTileNum;
        if (lastDimTileNum < sortTileInfo.maxCoreNum && sortTileInfo.unSortDimNum < int64_t(sortTileInfo.maxCoreNum)) {
            if (sortTileInfo.unSortDimNum < int64_t(bCore)) {
                bCore = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
                uint32_t hCore = sortTileInfo.maxCoreNum / bCore;
                uint32_t tileDataNew = Ops::Base::CeilDiv(int64_t(sortTileInfo.sortAxisNum), int64_t(hCore));
                tileData = Ops::Base::CeilAlign<uint32_t>(tileDataNew, BIN_NUM);
            }
        }
        if (bCore == static_cast<uint32_t>(1) && lastDimTileNum < sortTileInfo.maxCoreNum) {
            ComputeTileDataOne(sortTileInfo, lastDimTileNum, ubExtra, tileData, tileFactor);
        }
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return true;
    }
    return false;
}

uint32_t ComputeTileData(SortTileInfo &sortTileInfo)
{
    uint32_t indexSize;
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        indexSize = static_cast<uint32_t>(sizeof(int64_t));
    } else {
        indexSize = static_cast<uint32_t>(sizeof(int32_t));
    }
    uint32_t ubExtra = BIN_NUM * (indexSize + static_cast<uint32_t>(sizeof(uint16_t)) +
                                  static_cast<uint32_t>(sizeof(uint16_t)) + indexSize + indexSize);
    uint32_t tileFactor = sortTileInfo.dtypeSize + indexSize + static_cast<uint32_t>(sizeof(uint32_t)) +
                          static_cast<uint32_t>(sizeof(uint8_t)) + static_cast<uint32_t>(sizeof(uint8_t));

    uint32_t tileData = (sortTileInfo.ubSize - ubExtra) / tileFactor;
    tileData = (tileData / BIN_NUM) * BIN_NUM;

    uint32_t remainUb = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor);
    SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);

    uint32_t tmpUbSize = sortTileInfo.tmpUbSize;
    while (tmpUbSize > remainUb) {
        tileData = tileData - BIN_NUM;
        remainUb = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor);
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        tmpUbSize = sortTileInfo.tmpUbSize;
    }
    uint32_t lastDimTileNum = Ops::Base::CeilDiv(sortTileInfo.sortAxisNum, int64_t(tileData));
    OP_LOGI("RadixSortTiling", "tileData %u, lastDimTileNum %u, tmpUbSize %u", tileData, lastDimTileNum, tmpUbSize);
    bool smallTile =
        (sortTileInfo.sortAxisNum <= static_cast<int64_t>(SMALL_TILE_DATA_NUM)) && lastDimTileNum == uint32_t(1);
    if ((lastDimTileNum % sortTileInfo.maxCoreNum == static_cast<uint32_t>(0)) || smallTile) {
        OP_LOGI("RadixSortTiling", "lastDimTileNum align or smallTile");
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return tileData;
    }
    if (NeedAdjTileData(sortTileInfo, tileData, lastDimTileNum, ubExtra, tileFactor)) {
        return tileData;
    }
    AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
    return tileData;
}

ge::graphStatus GetSortMergeMultiCore(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    int64_t hCoreNum64 = (sortTileInfo.sortAxisNum + ONE_CORE_DATA_SIZE - 1) / ONE_CORE_DATA_SIZE;
    if (hCoreNum64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "sortAxisNum too large for merge multicore");
        return ge::GRAPH_FAILED;
    }
    uint32_t hCoreNum = static_cast<uint32_t>(hCoreNum64);
    if (sortTileInfo.sortAxisNum > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "sortAxisNum exceeds uint32_t limit");
        return ge::GRAPH_FAILED;
    }
    uint32_t tileNum = static_cast<uint32_t>(sortTileInfo.sortAxisNum) / hCoreNum;
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    sortTileInfo.lastDimNeedCore = hCoreNum;
    sortTileInfo.numTileDataSize = tileNum;
    uint64_t coreNumNeed64 = static_cast<uint64_t>(sortTileInfo.unSortDimNum) * static_cast<uint64_t>(hCoreNum);
    if (coreNumNeed64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "unSortDimNum * hCoreNum exceeds uint32_t limit");
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.coreNumNeed = static_cast<uint32_t>(coreNumNeed64);
    sortTileInfo.unsortedDimParallel = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
    sortTileInfo.sortLoopTimes = 1;

    // byteNum models per-element UB usage of merge sort pipeline:
    //   4lists * 8bytes * 2 (double buffer for input/output)
    //   + index buffer (uint32 or int64)
    //   + extract value buffer
    uint32_t byteNum = MERGE_SORT_DEALING_LIST_NUM * MERGE_SORT_DATASIZE * 2;
    byteNum += MERGE_SORT_DEALING_LIST_NUM * static_cast<uint32_t>(sizeof(uint32_t));
    if (sortTileInfo.y2DtypeSize == sizeof(int64_t)) {
        byteNum += MERGE_SORT_DEALING_LIST_NUM * static_cast<uint32_t>(sizeof(int64_t));
    }
    if (sortTileInfo.dataType == ge::DT_BF16) {
        byteNum += MERGE_SORT_DEALING_LIST_NUM * mergeType.find(ge::DT_BF16)->second;
        byteNum += MERGE_SORT_DEALING_LIST_NUM * static_cast<uint32_t>(sizeof(float));
    } else {
        byteNum += MERGE_SORT_DEALING_LIST_NUM * tilingDataTypeBitMap.find(sortTileInfo.dataType)->second;
    }
    sortTileInfo.keyParams0 = sortTileInfo.ubSize / byteNum;

    OP_LOGI("[mergeSort]", "maxDealingNum: %u", sortTileInfo.keyParams0);
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    size_t usrSize = static_cast<size_t>(
        MERGE_SORT_WORKSPACE_PARAM * sortTileInfo.sortAxisNum * sortTileInfo.unSortDimNum *
        static_cast<uint32_t>(sizeof(int32_t)));
    userWorkSpaceSize[0] = usrSize + WORK_SPACE_SIZE;
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetRadixSortOneCore(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    sortTileInfo.lastDimNeedCore = static_cast<uint32_t>(1);
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    uint64_t sortLoopTimes64 =
        Ops::Base::CeilDiv(sortTileInfo.unSortDimNum, static_cast<int64_t>(sortTileInfo.maxCoreNum));
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "sortLoopTimes exceeds uint32_t limit");
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    if (sortTileInfo.sortLoopTimes > static_cast<uint32_t>(1)) {
        sortTileInfo.coreNumNeed = sortTileInfo.maxCoreNum;
    } else {
        uint32_t core = static_cast<uint32_t>(sortTileInfo.unSortDimNum) % sortTileInfo.maxCoreNum;
        sortTileInfo.coreNumNeed = core == uint32_t(0) ? sortTileInfo.maxCoreNum : core;
    }
    sortTileInfo.unsortedDimParallel = sortTileInfo.coreNumNeed;
    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

void ComputeRadixMoreCoreWorkspace(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    uint32_t dtypeSizeWk = static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        dtypeSizeWk = static_cast<uint32_t>(sizeof(int64_t));
    }
    size_t excusiveBinsGmWkSize = static_cast<size_t>(sortTileInfo.keyParams1) * sortTileInfo.keyParams4 * dtypeSizeWk;
    excusiveBinsGmWkSize = Ops::Base::CeilAlign<size_t>(excusiveBinsGmWkSize,
        static_cast<size_t>(sortTileInfo.blockUbSize));

    size_t globalHistGmWkSize =
        static_cast<size_t>(sortTileInfo.keyParams3) * sortTileInfo.keyParams2 * sortTileInfo.keyParams0 * dtypeSizeWk;
    globalHistGmWkSize = Ops::Base::CeilAlign<size_t>(globalHistGmWkSize,
        static_cast<size_t>(sortTileInfo.blockUbSize));

    size_t outIdxDbWK = static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel * dtypeSizeWk;
    outIdxDbWK = Ops::Base::CeilAlign<size_t>(outIdxDbWK, static_cast<size_t>(sortTileInfo.blockUbSize));

    size_t histTileGmWk = static_cast<size_t>(sortTileInfo.lastDimTileNum) * BIN_NUM *
        sortTileInfo.unsortedDimParallel * sizeof(uint16_t) * 2U; // histTileGmWk and histCumsumTileGmWk

    size_t xB8GmWkSize = static_cast<size_t>(sortTileInfo.lastDimTileNum) * sortTileInfo.numTileDataSize *
        sortTileInfo.unsortedDimParallel;
    xB8GmWkSize = Ops::Base::CeilAlign<size_t>(xB8GmWkSize, static_cast<size_t>(sortTileInfo.blockUbSize));

    size_t outValueDbWKSize =
        static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel * sortTileInfo.dtypeSize;
    outValueDbWKSize = Ops::Base::CeilAlign<size_t>(outValueDbWKSize,
        static_cast<size_t>(sortTileInfo.blockUbSize));

    OP_LOGI("RadixSortTiling",
        "excusiveBinsGmWkSize %lu, globalHistGmWkSize %lu, outIdxDbWK %lu, histTileGmWk %lu,"
        " xB8GmWkSize %lu, outValueDbWKSize %lu ",
        excusiveBinsGmWkSize, globalHistGmWkSize, outIdxDbWK, histTileGmWk, xB8GmWkSize, outValueDbWKSize);
    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    size_t usrSize =
        excusiveBinsGmWkSize + globalHistGmWkSize + outIdxDbWK + histTileGmWk + xB8GmWkSize + outValueDbWKSize;
    userWorkSpaceSize[0] = usrSize + WORK_SPACE_SIZE;
    return;
}

// Fill tiling params for workspace clearing (globalHist + excusiveBins) in multi-core radix sort
static ge::graphStatus FillRadixMoreCoreClearParams(gert::TilingContext *context, SortTileInfo &sortTileInfo,
    uint32_t lastDimTileNum)
{
    uint32_t ubSizeNum = sortTileInfo.tmpUbSize / static_cast<uint32_t>(sizeof(uint32_t));
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        ubSizeNum = sortTileInfo.tmpUbSize / static_cast<uint32_t>(sizeof(int64_t));
    }
    uint64_t allNumGloblHist64 = static_cast<uint64_t>(BIN_NUM) * lastDimTileNum * sortTileInfo.dtypeSize *
        sortTileInfo.unsortedDimParallel;
    uint64_t allNumExcusiveBin64 = static_cast<uint64_t>(BIN_NUM) * sortTileInfo.dtypeSize *
        sortTileInfo.unsortedDimParallel;
    if (allNumGloblHist64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        allNumExcusiveBin64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "radix workspace tiling params exceed uint32_t limit");
        return ge::GRAPH_FAILED;
    }
    uint32_t allNumGloblHist = static_cast<uint32_t>(allNumGloblHist64);
    uint32_t allNumExcusiveBin = static_cast<uint32_t>(allNumExcusiveBin64);
    uint32_t oneCoreSize = Ops::Base::CeilDiv(int64_t(allNumGloblHist), int64_t(sortTileInfo.coreNumNeed));
    sortTileInfo.keyParams5 =
        std::max(static_cast<int64_t>(oneCoreSize), static_cast<int64_t>(sortTileInfo.blockUbSize));
    sortTileInfo.keyParams0 = Ops::Base::CeilDiv(int64_t(allNumGloblHist), int64_t(sortTileInfo.keyParams5));
    sortTileInfo.keyParams3 = Ops::Base::CeilDiv(int64_t(sortTileInfo.keyParams5), int64_t(ubSizeNum));
    sortTileInfo.keyParams2 = sortTileInfo.keyParams5 > ubSizeNum ? ubSizeNum : sortTileInfo.keyParams5;

    uint32_t oneCoreSize1 = Ops::Base::CeilDiv(int64_t(allNumExcusiveBin), int64_t(sortTileInfo.coreNumNeed));
    sortTileInfo.keyParams4 =
        std::max(static_cast<int64_t>(oneCoreSize1), static_cast<int64_t>(sortTileInfo.blockUbSize));

    sortTileInfo.keyParams1 = Ops::Base::CeilDiv(int64_t(allNumExcusiveBin), int64_t(sortTileInfo.keyParams4));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetRadixSortMoreCore(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    sortTileInfo.ubSize = sortTileInfo.ubSize - SIMT_UB;
    uint32_t tileData = ComputeTileData(sortTileInfo);
    if (tileData == 0) {
        OP_LOGE(context->GetNodeName(), "tileData is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t lastDimTileNum64 = (static_cast<uint64_t>(sortTileInfo.sortAxisNum) + tileData - 1U) / tileData;
    if (lastDimTileNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "lastDimTileNum exceeds uint32_t limit");
        return ge::GRAPH_FAILED;
    }
    uint32_t lastDimTileNum = static_cast<uint32_t>(lastDimTileNum64);
    if (sortTileInfo.maxCoreNum <= lastDimTileNum) {
        sortTileInfo.unsortedDimParallel = static_cast<uint32_t>(1);
    } else {
        sortTileInfo.unsortedDimParallel =
            lastDimTileNum == 0 ? sortTileInfo.maxCoreNum : sortTileInfo.maxCoreNum / lastDimTileNum;
        if (sortTileInfo.unSortDimNum < static_cast<int64_t>(sortTileInfo.unsortedDimParallel)) {
            sortTileInfo.unsortedDimParallel = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
        }
    }
    sortTileInfo.numTileDataSize = tileData;
    uint64_t sortLoopTimes64 = (static_cast<uint64_t>(sortTileInfo.unSortDimNum) +
        sortTileInfo.unsortedDimParallel - 1U) / sortTileInfo.unsortedDimParallel;
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "sortLoopTimes exceeds uint32_t limit");
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    sortTileInfo.lastDimNeedCore = std::min(sortTileInfo.maxCoreNum, lastDimTileNum);
    sortTileInfo.coreNumNeed = sortTileInfo.unsortedDimParallel * sortTileInfo.lastDimNeedCore;
    sortTileInfo.lastDimTileNum = lastDimTileNum;
    OP_CHECK_IF(FillRadixMoreCoreClearParams(context, sortTileInfo, lastDimTileNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "fill radix more-core clear params failed"), return ge::GRAPH_FAILED);
    ComputeRadixMoreCoreWorkspace(context, sortTileInfo);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void FillTilingDataSort(SortTileInfo &sortTileInfo, SortRegBaseTilingData *sortTilingData)
{
    sortTilingData->numTileDataSize = sortTileInfo.numTileDataSize;
    sortTilingData->unsortedDimParallel = sortTileInfo.unsortedDimParallel;
    sortTilingData->lastDimTileNum = sortTileInfo.lastDimTileNum;
    sortTilingData->sortLoopTimes = sortTileInfo.sortLoopTimes;
    sortTilingData->lastDimNeedCore = sortTileInfo.lastDimNeedCore;
    sortTilingData->keyParams0 = sortTileInfo.keyParams0;
    sortTilingData->keyParams1 = sortTileInfo.keyParams1;
    sortTilingData->keyParams2 = sortTileInfo.keyParams2;
    sortTilingData->keyParams3 = sortTileInfo.keyParams3;
    sortTilingData->keyParams4 = sortTileInfo.keyParams4;
    sortTilingData->keyParams5 = sortTileInfo.keyParams5;
    sortTilingData->tmpUbSize = sortTileInfo.tmpUbSize;
    sortTilingData->lastAxisNum = sortTileInfo.sortAxisNum;
    sortTilingData->unsortedDimNum = sortTileInfo.unSortDimNum;
    return;
}

void PrintTilindDataSort(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    OP_LOGI(context->GetNodeName(),
        "realCoreNum %u, numTileDataSize %u, unsortedDimParallel %u, "
        "lastDimTileNum %u, sortLoopTimes %u, lastDimNeedCore %u, keyParams0 %u, keyParams1 %u "
        "keyParams2 %u, keyParams3 %u, keyParams4 %u, keyParams5 %u, tmpUbSize %u, "
        "lastAxisNum %ld, unsortedDimNum %ld ",
        sortTileInfo.coreNumNeed, sortTileInfo.numTileDataSize, sortTileInfo.unsortedDimParallel,
        sortTileInfo.lastDimTileNum, sortTileInfo.sortLoopTimes, sortTileInfo.lastDimNeedCore, sortTileInfo.keyParams0,
        sortTileInfo.keyParams1, sortTileInfo.keyParams2, sortTileInfo.keyParams3, sortTileInfo.keyParams4,
        sortTileInfo.keyParams5, sortTileInfo.tmpUbSize, sortTileInfo.sortAxisNum, sortTileInfo.unSortDimNum);
    return;
}

struct MergeSortCorePlan {
    uint32_t alignNum = 0;
    uint32_t oneCoreRowNum = 0;
    uint32_t sortLoopTimes = 0;
    uint32_t coreNumNeed = 0;
};

static ge::graphStatus GetMergeSortCorePlan(gert::TilingContext *context, SortTileInfo &sortTileInfo,
    MergeSortCorePlan &plan)
{
    // 输入和排序结果 value 队列平分 tileData，计算单个核心一次最多处理的行数。
    uint32_t tileData = TILE_DATA_NUM;
    uint32_t oneCoreRowNumMax = (tileData / 2U) / plan.alignNum;
    oneCoreRowNumMax = (oneCoreRowNumMax == uint32_t(0) ? uint32_t(1) : oneCoreRowNumMax);
    if (sortTileInfo.unSortDimNum > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "unSortDimNum exceeds uint32_t limit");
        return ge::GRAPH_FAILED;
    }

    // 计算单轮分核，以及每个核心需要处理的 batch 数。
    uint32_t bUnsorted = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
    plan.coreNumNeed = std::min(bUnsorted, sortTileInfo.maxCoreNum);
    uint32_t batchPerCore = Ops::Base::CeilDiv(int64_t(bUnsorted), int64_t(plan.coreNumNeed));
    if (batchPerCore <= oneCoreRowNumMax) {
        plan.oneCoreRowNum = batchPerCore;
        plan.sortLoopTimes = 1U;
    } else {
        // batch 数超过单轮处理上限时，按 coreNumNeed * oneCoreRowNumMax 计算循环次数。
        plan.oneCoreRowNum = oneCoreRowNumMax;
        uint64_t totalRowsPerRound = static_cast<uint64_t>(plan.coreNumNeed) *
                                     static_cast<uint64_t>(plan.oneCoreRowNum);
        if (totalRowsPerRound == 0 ||
            totalRowsPerRound > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            OP_LOGE(context->GetNodeName(), "totalRowsPerRound exceeds int64_t limit");
            return ge::GRAPH_FAILED;
        }
        plan.sortLoopTimes = Ops::Base::CeilDiv(static_cast<int64_t>(bUnsorted),
            static_cast<int64_t>(totalRowsPerRound));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetMergeSort(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    MergeSortCorePlan plan;
    // 计算对齐后的排序轴元素数量。
    int64_t alignNum64 = Ops::Base::CeilAlign<int64_t>(sortTileInfo.sortAxisNum,
                                                       static_cast<int64_t>(sortTileInfo.blockUbSize));
    if (alignNum64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) || alignNum64 <= 0) {
        OP_LOGE(context->GetNodeName(), "sortAxisNum too large or alignNum invalid");
        return ge::GRAPH_FAILED;
    }
    plan.alignNum = static_cast<uint32_t>(alignNum64);
    OP_CHECK_IF(GetMergeSortCorePlan(context, sortTileInfo, plan) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get merge sort core plan failed"), return ge::GRAPH_FAILED);
    uint32_t maxTypeSize =
        (sortTileInfo.dataType == ge::DT_BF16) ? mergeType.find(ge::DT_FLOAT)->second : sortTileInfo.dtypeSize;
    auto platform_info = context->GetPlatformInfo();
    auto plat = platform_ascendc::PlatformAscendC(platform_info);
    // 计算 merge sort API 需要的临时 UB 大小。
    uint32_t dataSizeNeed = AscendC::GetConcatTmpSize(plat, plan.alignNum, maxTypeSize);
    sortTileInfo.tmpUbSize = std::max(dataSizeNeed, sortTileInfo.blockUbSize);

    // 填充 SortTileInfo 结构体。
    sortTileInfo.sortLoopTimes = plan.sortLoopTimes;
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    sortTileInfo.unsortedDimParallel = plan.coreNumNeed;
    sortTileInfo.lastDimNeedCore = static_cast<uint32_t>(1);
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    sortTileInfo.coreNumNeed = plan.coreNumNeed;
    sortTileInfo.keyParams0 = plan.oneCoreRowNum;
    sortTileInfo.keyParams1 = plan.alignNum * plan.oneCoreRowNum * sortTileInfo.dtypeSize;
    sortTileInfo.keyParams2 = plan.alignNum * plan.oneCoreRowNum * sortTileInfo.y2DtypeSize;
    sortTileInfo.keyParams3 = plan.alignNum;
    // 输入输出队列数量，超过2048无法开启double buffer。
    sortTileInfo.keyParams4 = sortTileInfo.sortAxisNum > 2048 ? 1 : 2;
    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief Compute workspace size for MergeIntraCore scenario
 * @details Workspace layout:
 *          [0]: TMP_CACHE - per-core cache for sorted blocks
 *          Cache stores sort struct format (8 bytes/element: value + index interleaved)
 *          Cache layout per core: 1 batch's ping-pong buffer (reused across batches)
 *          Each batch: GetSortLen(alignNum) * SORT_STRUCT_SIZE_FP32 bytes
 */
void ComputeMergeIntraCoreWorkSpace(gert::TilingContext *context, SortTileInfo &sortTileInfo,
    uint32_t batchPerCore, uint32_t alignNum)
{
    // Calculate cache size per batch using GetSortLen (32-element aligned)
    // Cache stores sort struct format (8 bytes/element: value + index interleaved)
    // Each batch needs alignNum * SORT_STRUCT_SIZE_FP32 * 2 (ping-pong)
    size_t cachePerBatch = static_cast<size_t>(alignNum) * SORT_STRUCT_SIZE_FP32 * 2;  // 8 bytes per element
    cachePerBatch = Ops::Base::CeilAlign<size_t>(cachePerBatch, static_cast<size_t>(sortTileInfo.blockUbSize));

    // Per core cache = 1 batch's cache (reused across batches within the same core)
    size_t cachePerCore = cachePerBatch;

    // Total workspace = per-core cache * actual core used
    uint32_t actualCoreNum = Ops::Base::CeilDiv(sortTileInfo.unSortDimNum, static_cast<int64_t>(batchPerCore));
    size_t totalCacheSize = cachePerCore * actualCoreNum;

    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = totalCacheSize + WORK_SPACE_SIZE;

    OP_LOGI("MergeIntraCoreTiling", "cachePerBatch %lu, cachePerCore %lu, actualCoreNum %u, totalCacheSize %lu",
        cachePerBatch, cachePerCore, actualCoreNum, totalCacheSize);
}

/**
 * @brief Get tiling parameters for MergeIntraCore scenario
 * @details Uses block-axis parallelism (batch dimension)
 *          Each core processes multiple batches serially
 */
void GetSortMergeIntraCore(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    // 1. Batch per core
    uint32_t batchPerCore = Ops::Base::CeilDiv(sortTileInfo.unSortDimNum,
        static_cast<int64_t>(sortTileInfo.maxCoreNum));
    sortTileInfo.keyParams0 = batchPerCore;

    // 2. Actual core number needed
    uint32_t actualCoreNum = Ops::Base::CeilDiv(sortTileInfo.unSortDimNum, static_cast<int64_t>(batchPerCore));
    sortTileInfo.coreNumNeed = actualCoreNum;
    sortTileInfo.unsortedDimParallel = actualCoreNum;

    // 3. Block sort size (elements per UB sort block)
    uint32_t blockSortSize = ComputeMergeIntraCoreBlockSortSize(sortTileInfo);

    // 4. Extract chunk size (elements per Phase 3 extract iteration)
    uint32_t extractChunkSize = ComputeMergeIntraCoreExtractChunkSize(sortTileInfo);

    sortTileInfo.numTileDataSize = blockSortSize;
    sortTileInfo.keyParams4 = extractChunkSize;

    // Calculate max merge iterations to prevent infinite loops: INT32_MAX / blockSortSize
    sortTileInfo.keyParams5 = (blockSortSize > 0) ?
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max() / blockSortSize) :
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max());

    // 5. Blocks per row (upper-bounded by 256 in IsSortMergeIntraCore)
    uint32_t blocksPerRow = Ops::Base::CeilDiv(sortTileInfo.sortAxisNum, static_cast<int64_t>(blockSortSize));
    sortTileInfo.lastDimTileNum = blocksPerRow;

    // 6. Aligned N for cache allocation
    uint32_t alignNum = blocksPerRow * blockSortSize;
    sortTileInfo.keyParams3 = alignNum;

    // 7. Actual core num used
    sortTileInfo.lastDimNeedCore = actualCoreNum;

    // 8. Compute workspace
    ComputeMergeIntraCoreWorkSpace(context, sortTileInfo, batchPerCore, alignNum);

    OP_LOGI("MergeIntraCoreTiling",
        "B %ld, N %ld, batchPerCore %u, actualCoreNum %u, blockSortSize %u, extractChunkSize %u, "
        "blocksPerRow %u, alignNum %u, ubSize %u",
        sortTileInfo.unSortDimNum, sortTileInfo.sortAxisNum, batchPerCore, actualCoreNum,
        blockSortSize, extractChunkSize, blocksPerRow, alignNum, sortTileInfo.ubSize);
}

bool TrySmallAxis(gert::TilingContext *context, SortTileInfo &sortTileInfo, uint64_t &schId)
{
    if (sortTileInfo.sortAxisNum > static_cast<int64_t>(SMALL_AXIS_THRESHOLD)) {
        return false;
    }
    if (IsAxisOneCopy(sortTileInfo)) {
        if (GetAxisOneCopy(context, sortTileInfo) != ge::GRAPH_SUCCESS) {
            return false;
        }
        schId = SORT_SCHID_7;
        return true;
    }
    SmallAxisRoutePlan smallAxisRoutePlan;
    if (!SelectSmallAxisRoute(sortTileInfo, smallAxisRoutePlan)) {
        return false;
    }
    if (smallAxisRoutePlan.kind == SmallAxisRouteKind::TWO_STAGE) {
        schId = SORT_SCHID_6;
    } else if (smallAxisRoutePlan.kind == SmallAxisRouteKind::INSERTION) {
        schId = SORT_SCHID_5;
    } else {
        return false;
    }
    FillSmallAxisBatched(context, sortTileInfo, smallAxisRoutePlan);
    return true;
}

bool TryMerge(gert::TilingContext *context, SortTileInfo &sortTileInfo, uint64_t &schId)
{
    if (IsMergeSort(sortTileInfo)) {
        SortTileInfo candidate = sortTileInfo;
        if (GetMergeSort(context, candidate) == ge::GRAPH_SUCCESS) {
            sortTileInfo = candidate;
            schId = (sortTileInfo.sortAxisNum <= SORT32_SMALL_AXIS_THRESHOLD) ?
                SORT_SCHID_8 : static_cast<uint64_t>(0);
            return true;
        }
    }

    if (IsSortMergeMultiCore(sortTileInfo)) {
        SortTileInfo candidate = sortTileInfo;
        if (GetSortMergeMultiCore(context, candidate) == ge::GRAPH_SUCCESS) {
            sortTileInfo = candidate;
            schId = SORT_SCHID_3;
            return true;
        }
    }

    return false;
}

bool TryRadixOneCore(gert::TilingContext *context, SortTileInfo &sortTileInfo, uint64_t &schId)
{
    SortTileInfo candidate = sortTileInfo;
    if (!IsRadixSortOneCore(candidate) ||
        GetRadixSortOneCore(context, candidate) != ge::GRAPH_SUCCESS) {
        return false;
    }
    sortTileInfo = candidate;
    schId = static_cast<uint64_t>(1);
    return true;
}

bool TryMergeIntraCore(gert::TilingContext *context, SortTileInfo &sortTileInfo, uint64_t &schId)
{
    if (!IsSortMergeIntraCore(sortTileInfo)) {
        return false;
    }
    GetSortMergeIntraCore(context, sortTileInfo);
    schId = SORT_SCHID_4;
    return true;
}

ge::graphStatus SelectSortSchedule(gert::TilingContext *context, SortTileInfo &sortTileInfo, uint64_t &schId)
{
    if (TrySmallAxis(context, sortTileInfo, schId) ||
        TryMerge(context, sortTileInfo, schId) ||
        TryRadixOneCore(context, sortTileInfo, schId) ||
        TryMergeIntraCore(context, sortTileInfo, schId)) {
        return ge::GRAPH_SUCCESS;
    }

    schId = SORT_SCHID_2;
    OP_CHECK_IF(GetRadixSortMoreCore(context, sortTileInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "radix more-core tiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RadixSortTiling(gert::TilingContext *context, int32_t maxCoreNum)
{
    SortRegBaseTilingData *sortTilingData{ nullptr };
    sortTilingData = context->GetTilingData<SortRegBaseTilingData>();
    OP_CHECK_IF(sortTilingData == nullptr,
        OP_LOGE(context->GetNodeName(), "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(sortTilingData, sizeof(SortRegBaseTilingData), 0, sizeof(SortRegBaseTilingData)) != EOK),
        OP_LOGE(context->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    SortTileInfo sortTileInfo;
    OP_CHECK_IF(SortCheckParams(context, sortTileInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check params failed"), return ge::GRAPH_FAILED);
    sortTileInfo.maxCoreNum = static_cast<uint32_t>(maxCoreNum);
    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    uint64_t isInt32 = static_cast<uint64_t>((sortTileInfo.sortAxisNum <= int32Max));
    const bool *isDescending = context->GetAttrs()->GetAttrPointer<bool>(1);
    uint64_t isDescend = *isDescending;
    sortTileInfo.isDescend = static_cast<bool>(isDescend);
    sortTileInfo.isInt32 = static_cast<uint32_t>(isInt32);
    OP_LOGI(context->GetNodeName(), "isInt32 is %lu, isDescend is %lu", isInt32, isDescend);
    uint64_t schId = static_cast<uint64_t>(0);
    OP_CHECK_IF(SelectSortSchedule(context, sortTileInfo, schId) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "select sort schedule failed"), return ge::GRAPH_FAILED);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, isInt32, isDescend);
    OP_LOGI(context->GetNodeName(), "tilingKey is %lu, maxCoreNum %d, schId %lu", tilingKey, maxCoreNum, schId);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(sortTileInfo.coreNumNeed);
    context->SetLocalMemorySize(sortTileInfo.ubSize);
    FillTilingDataSort(sortTileInfo, sortTilingData);
    PrintTilindDataSort(context, sortTileInfo);
    OP_LOGI(context->GetNodeName(), "end RadixSortTIling ");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortTilingSimt(gert::TilingContext *context, int32_t maxCoreNum)
{
    return RadixSortTiling(context, maxCoreNum);
}
}
