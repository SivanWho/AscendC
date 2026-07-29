/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_SMALL_AXIS_INSERTION_H
#define SORT_SMALL_AXIS_INSERTION_H

#include "basic_api/kernel_vec_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "sort_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "util_type_simd.h"

namespace Sort {
using namespace AscendC;

constexpr uint32_t INSERTION_THREAD_NUM = 256;
// DataCopy hardware limit for blockCount (batch mode max segments per transfer).
constexpr uint32_t MAX_DATACOPY_BLOCK_COUNT = 4095;

template <typename T, typename IDX_T, bool IsDescend>
__simt_vf__ LAUNCH_BOUND(INSERTION_THREAD_NUM) __aicore__
void SimtInsertionSortSegments(uint32_t validSegs, uint32_t segmentLen,
    uint32_t valueRowElems, uint32_t idxRowElems,
    __ubuf__ T       *valueBase,
    __ubuf__ IDX_T   *idxBase)
{
    for (int32_t seg = threadIdx.x; seg < static_cast<int32_t>(validSegs);
         seg += static_cast<int32_t>(INSERTION_THREAD_NUM)) {
        uint32_t segId = static_cast<uint32_t>(seg);
        uint32_t valueBaseOffset = segId * valueRowElems;
        uint32_t idxBaseOffset = segId * idxRowElems;
        for (uint32_t elem = 0; elem < segmentLen; ++elem) {
            idxBase[idxBaseOffset + elem] = static_cast<IDX_T>(elem);
        }

        for (uint32_t elem = 1; elem < segmentLen; ++elem) {
            T keyValue = valueBase[valueBaseOffset + elem];
            IDX_T keyIdx = idxBase[idxBaseOffset + elem];
            uint32_t insertPos = elem;
            while (insertPos > 0) {
                T prevValue = valueBase[valueBaseOffset + insertPos - 1];
                bool needMove;
                if constexpr (IsDescend) {
                    needMove = prevValue < keyValue;
                } else {
                    needMove = prevValue > keyValue;
                }
                if (!needMove) {
                    break;
                }
                valueBase[valueBaseOffset + insertPos] = prevValue;
                idxBase[idxBaseOffset + insertPos] = idxBase[idxBaseOffset + insertPos - 1];
                --insertPos;
            }
            valueBase[valueBaseOffset + insertPos] = keyValue;
            idxBase[idxBaseOffset + insertPos] = keyIdx;
        }
    }
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
class SortSmallAxisInsertion {
public:
    __aicore__ inline SortSmallAxisInsertion() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
        const SortRegBaseTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    // Main pipeline functions (in call order)
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void LoadBatch(int64_t segStart, uint32_t validSegs);
    __aicore__ inline void ProcessSegmentInsertionSort(uint32_t validSegs);
    __aicore__ inline void StoreBatch(int64_t segStart, uint32_t validSegs);

    TPipe *pipe_ = nullptr;
    const SortRegBaseTilingData *tilingData_ = nullptr;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<OUT_IDX_T> outIdxGm_;

    TBuf<TPosition::VECCALC> valueBuf_;       // CONVERT_TYPE buffer for sort operations
    TBuf<TPosition::VECCALC> castBuf_;        // T buffer for GM ↔ UB cast (bf16↔float transfer)
    TBuf<TPosition::VECCALC> idxBuf_;

    LocalTensor<CONVERT_TYPE> values_;
    LocalTensor<OUT_IDX_T> indices_;

    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t segmentLen_ = 0;
    uint32_t segmentsPerBatch_ = 0;
    uint32_t batchNum_ = 0;
    int64_t totalSegs_ = 0;
    uint32_t valueRowStride_ = 0;
    uint32_t castRowStride_ = 0;
    uint32_t indexRowStride_ = 0;
};

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace, const SortRegBaseTilingData *tilingData, TPipe *pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    pipe_ = pipe;
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    segmentLen_ = tilingData_->numTileDataSize;
    segmentsPerBatch_ = tilingData_->keyParams0;
    batchNum_ = tilingData_->keyParams1;
    totalSegs_ = tilingData_->unsortedDimNum;
    indexRowStride_ = ROUND_UP_AGLIN(segmentLen_ * sizeof(OUT_IDX_T)) / sizeof(OUT_IDX_T);
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        castRowStride_ = ROUND_UP_AGLIN(segmentLen_ * sizeof(T)) / sizeof(T);
        valueRowStride_ = castRowStride_;
    } else {
        valueRowStride_ = ROUND_UP_AGLIN(segmentLen_ * sizeof(CONVERT_TYPE)) / sizeof(CONVERT_TYPE);
    }

    inputXGm_.SetGlobalBuffer((__gm__ T *)x);
    outValueGm_.SetGlobalBuffer((__gm__ T *)y);
    outIdxGm_.SetGlobalBuffer((__gm__ OUT_IDX_T *)idx);

    if (segmentLen_ == 0 || segmentsPerBatch_ == 0 || segmentsPerBatch_ > MAX_DATACOPY_BLOCK_COUNT) {
        return;
    }

    pipe_->InitBuffer(valueBuf_, segmentsPerBatch_ * valueRowStride_ * sizeof(CONVERT_TYPE));
    pipe_->InitBuffer(idxBuf_, segmentsPerBatch_ * indexRowStride_ * sizeof(OUT_IDX_T));
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        pipe_->InitBuffer(castBuf_, segmentsPerBatch_ * castRowStride_ * sizeof(T));
    }

    values_ = valueBuf_.Get<CONVERT_TYPE>();
    indices_ = idxBuf_.Get<OUT_IDX_T>();
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline uint32_t SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::ComputeValidSegs(
    uint32_t batchId) const
{
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
    int64_t segRemain = totalSegs_ - segStart;
    if (segRemain <= 0) {
        return 0;
    }
    if (segRemain >= static_cast<int64_t>(segmentsPerBatch_)) {
        return segmentsPerBatch_;
    }
    return static_cast<uint32_t>(segRemain);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::LoadBatch(int64_t segStart,
    uint32_t validSegs)
{
    if (validSegs == 0) {
        return;
    }
    int64_t gmOffset = segStart * segmentLen_;
    DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
    // For diff-type, valueRowStride_ is intentionally equal to castRowStride_, so the
    // GM-transfer T buffer and the CONVERT_TYPE sort buffer share the same row pitch.
    uint32_t ubStrideBlocks = ((valueRowStride_ - segmentLen_) * sizeof(T)) / UB_BLOCK_SIZE;
    DataCopyExtParams copyParam{
        static_cast<uint16_t>(validSegs), static_cast<uint32_t>(segmentLen_ * sizeof(T)), 0, ubStrideBlocks, 0 };
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        LocalTensor<T> loadBuf = castBuf_.Get<T>();
        DataCopyPad(loadBuf, inputXGm_[gmOffset], copyParam, padParams);
        // MTE2 writes loadBuf; vector Cast consumes it.
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Cast(values_, loadBuf, RoundMode::CAST_NONE, validSegs * valueRowStride_);
        // Cast writes values_; VF insertion sort consumes it.
    } else {
        DataCopyPad(values_, inputXGm_[gmOffset], copyParam, padParams);
        // MTE2 writes values_; SIMT/VF insertion sort consumes it.
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::ProcessSegmentInsertionSort(
    uint32_t validSegs)
{
    asc_vf_call<SimtInsertionSortSegments<CONVERT_TYPE, OUT_IDX_T, IsDescend>>(
        dim3(INSERTION_THREAD_NUM),
        validSegs, segmentLen_,
        valueRowStride_, indexRowStride_,
        (__ubuf__ CONVERT_TYPE *)values_.GetPhyAddr(),
        (__ubuf__ OUT_IDX_T *)indices_.GetPhyAddr());
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::StoreBatch(int64_t segStart,
    uint32_t validSegs)
{
    if (validSegs == 0) {
        return;
    }
    // SIMT/VF insertion sort writes values_/indices_; store path consumes those UB buffers.
    event_t e = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);

    int64_t gmOffset = segStart * segmentLen_;
    // For diff-type, valueRowStride_ is intentionally equal to castRowStride_, so the
    // GM-transfer T buffer and the CONVERT_TYPE sort buffer share the same row pitch.
    uint32_t valueSrcStrideBlocks = ((valueRowStride_ - segmentLen_) * sizeof(T)) / UB_BLOCK_SIZE;
    DataCopyExtParams valueCopyParam{
        static_cast<uint16_t>(validSegs), static_cast<uint32_t>(segmentLen_ * sizeof(T)), valueSrcStrideBlocks, 0, 0 };
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        LocalTensor<T> storeBuf = castBuf_.Get<T>();
        Cast(storeBuf, values_, RoundMode::CAST_RINT, validSegs * valueRowStride_);
        // Cast writes storeBuf on vector pipe; MTE3 DataCopyPad reads it for GM writeback.
        event_t e = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
        DataCopyPad(outValueGm_[gmOffset], storeBuf, valueCopyParam);
    } else {
        DataCopyPad(outValueGm_[gmOffset], values_, valueCopyParam);
    }

    uint32_t idxStrideBlocks = ((indexRowStride_ - segmentLen_) * sizeof(OUT_IDX_T)) / UB_BLOCK_SIZE;
    DataCopyExtParams idxCopyParam{
        static_cast<uint16_t>(validSegs),
        static_cast<uint32_t>(segmentLen_ * sizeof(OUT_IDX_T)),
        idxStrideBlocks,
        0,
        0 };
    DataCopyPad(outIdxGm_[gmOffset], indices_, idxCopyParam);

    // The next batch is issued by scalar/control flow after these MTE3 writebacks complete.
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventId);
    WaitFlag<HardEvent::MTE3_S>(eventId);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::Process()
{
    if (blockIdx_ >= blockDim_ || segmentLen_ == 0 || segmentsPerBatch_ == 0) {
        return;
    }

    uint32_t batchesPerCore = (batchNum_ + blockDim_ - 1) / blockDim_;
    uint32_t startBatch = blockIdx_ * batchesPerCore;
    uint32_t endBatch = batchNum_ < startBatch + batchesPerCore ? batchNum_ : startBatch + batchesPerCore;

    for (uint32_t batchId = startBatch; batchId < endBatch; ++batchId) {
        uint32_t validSegs = ComputeValidSegs(batchId);
        if (validSegs == 0) {
            continue;
        }
        int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
        LoadBatch(segStart, validSegs);
        ProcessSegmentInsertionSort(validSegs);
        StoreBatch(segStart, validSegs);
    }
}

} // namespace Sort

#endif
