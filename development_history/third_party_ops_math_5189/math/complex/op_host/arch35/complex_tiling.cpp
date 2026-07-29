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
 * \file    complex_tiling.cpp
 * \brief   Complex operator tiling implementation
 */

#include "complex_tiling.h"
#include <graph/utils/type_utils.h>

using namespace ge;

namespace optiling {

// ---------------------------------------------------------------------------
// Tuning knobs
// ---------------------------------------------------------------------------
namespace {
constexpr int64_t kSimtGridDimMax = 56 * 16;
constexpr int64_t kSimtBlockDimMax = 2048;
constexpr int64_t kElementsPerThreadMax = 2048;
constexpr int64_t kElementsPerThreadList[] = {1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048};
constexpr bool kAllowGridDimLessThanCoreNumForSmallN = true;
constexpr bool kAlignBlockDimToWarp = false;
constexpr int64_t kWarpSize = 32;
}  // namespace

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline int64_t AlignUp(int64_t x, int64_t align)
{
    if (align == 0) {
        return x;
    } else  {
        return ((x + align - 1) / align) * align;
    }
}

// ---------------------------------------------------------------------------
// Input / output indices
// ---------------------------------------------------------------------------
static constexpr uint64_t INPUT_REAL = 0;
static constexpr uint64_t INPUT_IMAG = 1;
static constexpr uint64_t OUTPUT_OUT = 0;

// ---------------------------------------------------------------------------
// GetPlatformInfo
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "ComplexTiling GetPlatformInfo.");
    compileInfo_ = static_cast<const ComplexCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckDtype
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::CheckDtype()
{
    OP_LOGD(context_->GetNodeName(), "ComplexTiling CheckDtype.");

    auto realDesc = context_->GetInputDesc(INPUT_REAL);
    OP_CHECK_NULL_WITH_CONTEXT(context_, realDesc);
    ge::DataType realDtype = realDesc->GetDataType();

    auto imagDesc = context_->GetInputDesc(INPUT_IMAG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, imagDesc);
    ge::DataType imagDtype = imagDesc->GetDataType();

    auto outDesc = context_->GetOutputDesc(OUTPUT_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);
    ge::DataType outDtype = outDesc->GetDataType();

    if (realDtype != imagDtype) {
        std::string realStr = ge::TypeUtils::DataTypeToSerialString(realDtype);
        std::string imagStr = ge::TypeUtils::DataTypeToSerialString(imagDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "real and imag",
            (realStr + ", " + imagStr).c_str(),
            "The dtypes of real and imag must be identical (both float32 or both float16)");
        return ge::GRAPH_FAILED;
    }

    bool valid = false;
    if (realDtype == ge::DT_FLOAT && outDtype == ge::DT_COMPLEX64) {
        valid = true;
    } else if (realDtype == ge::DT_FLOAT16 && outDtype == ge::DT_COMPLEX32) {
        valid = true;
    }

    if (!valid) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(realDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(outDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "real/imag, out",
            dtypesStr.c_str(),
            "Expected float32->complex64 or float16->complex32");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckBroadcastAndMergeShape
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::CheckBroadcastAndMergeShape()
{
    OP_LOGD(context_->GetNodeName(), "ComplexTiling CheckBroadcastAndMergeShape.");

    const gert::StorageShape* realStorageShape = context_->GetInputShape(INPUT_REAL);
    OP_CHECK_NULL_WITH_CONTEXT(context_, realStorageShape);
    const gert::StorageShape* imagStorageShape = context_->GetInputShape(INPUT_IMAG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, imagStorageShape);

    auto& realShape = realStorageShape->GetStorageShape();
    auto& imagShape = imagStorageShape->GetStorageShape();

    int64_t realDimNum = static_cast<int64_t>(realShape.GetDimNum());
    int64_t imagDimNum = static_cast<int64_t>(imagShape.GetDimNum());
    dimNum_ = std::max(realDimNum, imagDimNum);
    OP_CHECK_IF(dimNum_ > COMPLEX_MAX_DIM,
        OP_LOGE(context_, "dimNum %ld exceeds COMPLEX_MAX_DIM %ld", dimNum_, static_cast<int64_t>(COMPLEX_MAX_DIM)),
        return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < dimNum_; i++) {
        int64_t realOffset = i - (dimNum_ - realDimNum);
        int64_t imagOffset = i - (dimNum_ - imagDimNum);

        // Read raw dims (int64_t from framework), validate, then convert to uint64_t
        int64_t realDimRaw = (realOffset >= 0) ? realShape.GetDim(realOffset) : 1;
        int64_t imagDimRaw = (imagOffset >= 0) ? imagShape.GetDim(imagOffset) : 1;

        OP_CHECK_IF(realDimRaw <= 0,
            OP_LOGE(context_, "real dim[%ld] is %ld, all dims must be > 0 (empty / dynamic shape not supported)",
                    realOffset, realDimRaw),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(imagDimRaw <= 0,
            OP_LOGE(context_, "imag dim[%ld] is %ld, all dims must be > 0 (empty / dynamic shape not supported)",
                    imagOffset, imagDimRaw),
            return ge::GRAPH_FAILED);

        uint64_t realDim = static_cast<uint64_t>(realDimRaw);
        uint64_t imagDim = static_cast<uint64_t>(imagDimRaw);

        realDims_[i] = realDim;
        imagDims_[i] = imagDim;

        OP_CHECK_IF(realDim != imagDim && realDim != 1 && imagDim != 1,
            OP_LOGE(context_, "Shapes not broadcastable at dim %ld: %lu vs %lu", i, realDim, imagDim),
            return ge::GRAPH_FAILED);
        mergedShape_[i] = std::max(realDim, imagDim);
    }

    totalElements_ = 1;
    for (int64_t i = 0; i < dimNum_; i++) {
        totalElements_ *= mergedShape_[i];
    }

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// MergeDims
//   合并相邻可合并维度,减少 kernel 中 broadcast 寻址的除法次数。
//
//   原理:
//   1) 先 squeeze 掉 mergedShape==1 的维度(它们对寻址无贡献);
//   2) 相邻两维 i / i+1 可合并 当且仅当 real、imag 各自的 broadcast 状态相同:
//        (realDims[i]==1) == (realDims[i+1]==1)   且
//        (imagDims[i]==1) == (imagDims[i+1]==1)
//      —— 此时两维要么都连续、要么都广播,合并后仍可用单一线性 stride 表达。
//   合并后直接改写 mergedShape_/realDims_/imagDims_/dimNum_,
//   随后的 CalcStride 会基于压缩维度重算正确 stride,无需改动。
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::MergeDims()
{
    uint64_t mShape[COMPLEX_MAX_DIM];
    uint64_t rDim[COMPLEX_MAX_DIM];
    uint64_t iDim[COMPLEX_MAX_DIM];
    int64_t n = 0;

    // Step 1: squeeze 掉 merged size == 1 的维度
    for (int64_t i = 0; i < dimNum_; i++) {
        if (mergedShape_[i] == 1) {
            continue;
        }
        mShape[n] = mergedShape_[i];
        rDim[n]   = realDims_[i];
        iDim[n]   = imagDims_[i];
        n++;
    }

    // 全部为 1(标量化场景)时,保留一个 size=1 的维度
    if (n == 0) {
        mShape[0] = 1;
        rDim[0]   = 1;
        iDim[0]   = 1;
        n = 1;
    }

    // Step 2: 合并相邻可合并维度
    int64_t out = 0;
    for (int64_t i = 1; i < n; i++) {
        bool realSame = ((rDim[out] == 1) == (rDim[i] == 1));
        bool imagSame = ((iDim[out] == 1) == (iDim[i] == 1));
        if (realSame && imagSame) {
            mShape[out] *= mShape[i];
            rDim[out]   *= rDim[i];   // 都广播: 1*1=1; 都连续: 乘积
            iDim[out]   *= iDim[i];
        } else {
            out++;
            mShape[out] = mShape[i];
            rDim[out]   = rDim[i];
            iDim[out]   = iDim[i];
        }
    }
    int64_t newDim = out + 1;

    // 写回压缩后的维度信息
    dimNum_ = newDim;
    for (int64_t i = 0; i < newDim; i++) {
        mergedShape_[i] = mShape[i];
        realDims_[i]    = rDim[i];
        imagDims_[i]    = iDim[i];
    }
    for (int64_t i = newDim; i < COMPLEX_MAX_DIM; i++) {
        mergedShape_[i] = 1;
        realDims_[i]    = 1;
        imagDims_[i]    = 1;
    }

    OP_LOGD(context_->GetNodeName(), "ComplexTiling MergeDims: dimNum %ld -> %ld", static_cast<int64_t>(dimNum_), newDim);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CalcStride
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::CalcStride()
{
    OP_LOGD(context_->GetNodeName(), "ComplexTiling CalcStride.");

    uint64_t strideReal = 1;
    uint64_t strideImag = 1;
    uint64_t strideMerged = 1;

    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        realStride_[i]   = (realDims_[i] == 1) ? 0 : strideReal;
        imagStride_[i]   = (imagDims_[i] == 1) ? 0 : strideImag;
        mergedStride_[i] = strideMerged;
        strideReal   *= realDims_[i];
        strideImag   *= imagDims_[i];
        strideMerged *= mergedShape_[i];
    }

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// SearchSimtConfig
//   搜索合法的 (gridDim, blockDim, elementsPerThread) 三元组。
//   遍历 elements-per-thread 候选列表与 wave 数量，在硬件上限约束下
//   寻找第一个能满足 blockDim ≤ kSimtBlockDimMax 的配置。
//   若未找到任何合法配置，回退到最大上限。
// ---------------------------------------------------------------------------
void ComplexTiling::SearchSimtConfig(uint64_t N, int64_t C,
                                      int64_t& gridDim, int64_t& blockDim,
                                      int64_t& elementsPerThread)
{
    gridDim = 1;
    blockDim = 1;
    elementsPerThread = 1;

    bool found = false;
    for (int64_t E : kElementsPerThreadList) {
        for (int64_t waves = 1; ; ++waves) {
            int64_t rawGridDim = waves * C;
            if (rawGridDim > kSimtGridDimMax) break;

            int64_t G = rawGridDim;
            if (kAllowGridDimLessThanCoreNumForSmallN &&
                static_cast<uint64_t>(N) < static_cast<uint64_t>(G)) {
                G = static_cast<int64_t>(N);
            }
            if (G <= 0) G = 1;

            uint64_t maxElemsPerBlock =
                (N + static_cast<uint64_t>(G) - 1) / static_cast<uint64_t>(G);
            uint64_t BU =
                (maxElemsPerBlock + static_cast<uint64_t>(E) - 1) / static_cast<uint64_t>(E);
            if (BU < 1) BU = 1;

            int64_t B = static_cast<int64_t>(BU);
            if (kAlignBlockDimToWarp && B > 1) {
                B = AlignUp(B, kWarpSize);
            }

            if (B <= kSimtBlockDimMax) {
                gridDim = G;
                blockDim = B;
                elementsPerThread = E;
                found = true;
                break;
            }

            if (G == static_cast<int64_t>(N) && kAllowGridDimLessThanCoreNumForSmallN) break;
        }
        if (found) break;
    }

    if (!found) {
        gridDim = kSimtGridDimMax;
        blockDim = kSimtBlockDimMax;
        elementsPerThread = kElementsPerThreadMax;
    }
}

// ---------------------------------------------------------------------------
// FillStridesAndMeta
//   填充 tilingData_ 中的 stride 数组、执行模式与 dtype 标记。
//   stride: 将 mergedStride_/realStride_/imagStride_ 拷贝至 tilingData_,
//           超出 dimNum_ 的维度填默认值。
//   mode:   检查是否存在 broadcast，无 broadcast 时启用 FAST_CONTIGUOUS 快速路径。
//   dtype:  根据 real 输入的 dtype 设置标记位(float16=1, float32=0)。
// ---------------------------------------------------------------------------
void ComplexTiling::FillStridesAndMeta()
{
    for (int64_t i = 0; i < COMPLEX_MAX_DIM; i++) {
        if (i < dimNum_) {
            tilingData_.mergedStride[i] = mergedStride_[i];
            tilingData_.realStride[i] = realStride_[i];
            tilingData_.imagStride[i] = imagStride_[i];
        } else {
            tilingData_.mergedStride[i] = 1;
            tilingData_.realStride[i] = 0;
            tilingData_.imagStride[i] = 0;
        }
    }

    bool noBroadcast = true;
    for (int64_t i = 0; i < dimNum_; i++) {
        if (realDims_[i] != mergedShape_[i] || imagDims_[i] != mergedShape_[i]) {
            noBroadcast = false;
            break;
        }
    }
    tilingData_.mode = noBroadcast ? MODE_FAST_CONTIGUOUS : MODE_GENERAL_BROADCAST;

    auto realDesc = context_->GetInputDesc(INPUT_REAL);
    ge::DataType realDtype = realDesc->GetDataType();
    tilingData_.dtype = (realDtype == ge::DT_FLOAT16) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// ComputeSimtConfigAndFill
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::ComputeSimtConfigAndFill()
{
    OP_LOGD(context_->GetNodeName(), "ComplexTiling ComputeSimtConfigAndFill.");

    const uint64_t N = totalElements_;
    if (N == 0) {
        OP_LOGE(context_, "totalElements is 0");
        return ge::GRAPH_FAILED;
    }

    const int64_t C = compileInfo_->coreNum;
    int64_t gridDim = 1;
    int64_t blockDim = 1;
    int64_t elementsPerThread = 1;
    SearchSimtConfig(N, C, gridDim, blockDim, elementsPerThread);

    uint64_t elementsPerBlock = N / static_cast<uint64_t>(gridDim);
    uint64_t formerBlock = N % static_cast<uint64_t>(gridDim);

    tilingData_.totalElements = static_cast<int64_t>(N);
    tilingData_.gridDim = static_cast<int64_t>(gridDim);
    tilingData_.blockDim = static_cast<int64_t>(blockDim);
    tilingData_.elementsPerThread = static_cast<int64_t>(elementsPerThread);
    tilingData_.elementsPerBlock = static_cast<int64_t>(elementsPerBlock);
    tilingData_.formerBlock = static_cast<int64_t>(formerBlock);
    tilingData_.dimNum = dimNum_;

    FillStridesAndMeta();

    OP_LOGD(context_->GetNodeName(),
            "ComplexTiling SIMT config: CoreNum=%lld G(gridDim)=%ld B(blockDim)=%ld "
            "E(elementsPerThread)=%ld ePerBlock=%lu former=%lu mode=%d dtype=%d",
            compileInfo_->coreNum,
            gridDim, blockDim, elementsPerThread, elementsPerBlock, formerBlock,
            tilingData_.mode, tilingData_.dtype);

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// GetShapeAttrsInfo
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::GetShapeAttrsInfo()
{
    if (CheckDtype() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (CheckBroadcastAndMergeShape() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (MergeDims() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;   // <-- 新增
    if (CalcStride() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// DoOpTiling
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "ComplexTiling DoOpTiling.");
    return ComputeSimtConfigAndFill();
}

// ---------------------------------------------------------------------------
// PostTiling
// ---------------------------------------------------------------------------
ge::graphStatus ComplexTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "ComplexTiling PostTiling.");

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0;

    uint32_t gridDim = static_cast<uint32_t>(tilingData_.gridDim);
    auto res = context_->SetBlockDim(gridDim);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        &tilingData_, sizeof(ComplexTilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(ComplexTilingData));

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// Tiling registration entry points
// ---------------------------------------------------------------------------
static ge::graphStatus Tiling4Complex(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4Complex start.");

    ComplexTiling complexTiling(context);
    auto ret = complexTiling.DoTiling();
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGE(context->GetNodeName(), "Tiling4Complex failed!"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Tiling4Complex end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ComplexAscendc(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4ComplexAscendc.");

    auto compileInfo = context->GetCompiledInfo<ComplexCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "core num is negative."), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4ComplexAscendc.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Complex(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ComplexCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("TilingPrepare4Complex", "Ascend C TilingPrepare4Complex success.");
    return TilingPrepare4ComplexAscendc(context);
}

IMPL_OP_OPTILING(Complex).Tiling(Tiling4Complex).TilingParse<ComplexCompileInfo>(TilingPrepare4Complex);

}  // namespace optiling
