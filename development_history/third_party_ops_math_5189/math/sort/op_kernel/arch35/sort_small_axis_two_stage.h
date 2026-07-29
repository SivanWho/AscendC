/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_SMALL_AXIS_TWO_STAGE_H
#define SORT_SMALL_AXIS_TWO_STAGE_H

#include "basic_api/kernel_vec_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "sort_tiling_data.h"
#include "simt_api/asc_simt.h"

namespace Sort {
using namespace AscendC;

constexpr uint32_t TWO_STAGE_THREAD_NUM = 1024;

// Scatter sorted values back to per-segment order via rank-inverse mapping.
// Example (2 segments x 3 elements, ascend sort):
//   Input: [[3, 1, 5], [4, 2, 6]] -> flatten [3, 1, 5, 4, 2, 6]
//   Stage1 sorts flatten: stage1Values=[1, 2, 3, 4, 5, 6], stage1Order=[1, 4, 0, 3, 2, 5]
//   Build rankInverse: rankInverse[flatIdx]=rank -> [2, 0, 4, 3, 1, 5]
//   Scatter to per-seg: flatIdx 0(rank=2)->seg0[1]=3, flatIdx 1(rank=0)->seg0[0]=1, ...
//   Final: values=[1, 3, 5, 2, 4, 6], indices=[1, 0, 2, 1, 0, 2]
template <typename T, typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(TWO_STAGE_THREAD_NUM) __aicore__
void RankInverseScatter(uint32_t totalElems, uint32_t segmentLen,
    __ubuf__ T *stage1ValuePtr, __ubuf__ uint32_t *stage1OrderPtr, __ubuf__ uint32_t *rankInversePtr,
    __ubuf__ T *finalValuePtr, __ubuf__ OutIdxT *finalIdxPtr)
{
    // Build inverse mapping: for each flattened index, store its rank from stage-1 sort.
    // rankInverse[stage1Order[rank]] = rank, e.g., stage1Order=[1, 4, 0, 3, 2, 5] -> rankInverse=[2, 0, 4, 3, 1, 5].
    for (uint32_t rank = static_cast<uint32_t>(threadIdx.x); rank < totalElems; rank += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = stage1OrderPtr[rank];
        rankInversePtr[flatIdx] = rank;
    }
    asc_syncthreads();

    // Compute final position for each element by counting how many same-segment elements have lower rank.
    for (uint32_t flat = static_cast<uint32_t>(threadIdx.x); flat < totalElems; flat += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = flat;
        uint32_t segId = flatIdx / segmentLen;
        uint32_t laneId = flatIdx - segId * segmentLen;

        uint32_t rank = rankInversePtr[flatIdx];
        uint32_t segBase = segId * segmentLen;
        // 8-way unroll: reduces loop overhead and hides memory latency via independent accumulators.
        uint32_t pos0 = 0;
        uint32_t pos1 = 0;
        uint32_t pos2 = 0;
        uint32_t pos3 = 0;
        uint32_t pos4 = 0;
        uint32_t pos5 = 0;
        uint32_t pos6 = 0;
        uint32_t pos7 = 0;
        uint32_t lane = 0;
        for (; lane + 7U < segmentLen; lane += 8U) {
            uint32_t scanBase = segBase + lane;
            pos0 += (rankInversePtr[scanBase] < rank) ? 1U : 0U;
            pos1 += (rankInversePtr[scanBase + 1U] < rank) ? 1U : 0U;
            pos2 += (rankInversePtr[scanBase + 2U] < rank) ? 1U : 0U;
            pos3 += (rankInversePtr[scanBase + 3U] < rank) ? 1U : 0U;
            pos4 += (rankInversePtr[scanBase + 4U] < rank) ? 1U : 0U;
            pos5 += (rankInversePtr[scanBase + 5U] < rank) ? 1U : 0U;
            pos6 += (rankInversePtr[scanBase + 6U] < rank) ? 1U : 0U;
            pos7 += (rankInversePtr[scanBase + 7U] < rank) ? 1U : 0U;
        }
        uint32_t pos = pos0 + pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + pos7;
        for (; lane < segmentLen; ++lane) {
            pos += (rankInversePtr[segBase + lane] < rank) ? 1U : 0U;
        }

        uint32_t outOffset = segBase + pos;
        finalValuePtr[outOffset] = stage1ValuePtr[rank];
        finalIdxPtr[outOffset] = static_cast<OutIdxT>(laneId);
    }
}


// Build packed keys for stage-2 sort: key = segId * totalElems + rank.
template <typename T, typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(TWO_STAGE_THREAD_NUM) __aicore__
void BuildStage2KeysSimt(uint32_t totalElems, uint32_t segmentLen,
    __ubuf__ uint32_t *stage1OrderPtr, __ubuf__ uint32_t *stage2KeyPtr)
{
    for (uint32_t rank = static_cast<uint32_t>(threadIdx.x); rank < totalElems; rank += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = stage1OrderPtr[rank];
        uint32_t segId = flatIdx / segmentLen;
        uint32_t laneId = flatIdx - segId * segmentLen;
        stage2KeyPtr[rank] = segId * totalElems + rank;
        stage1OrderPtr[rank] = laneId;
    }
}

template <typename T, typename OutIdxT, bool IsDescend>
class SortSmallAxisTwoStage {
public:
    __aicore__ inline SortSmallAxisTwoStage() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
        const SortRegBaseTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t AlignUpBytes(uint32_t bytes) const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void LoadBatch(int64_t segStart, uint32_t totalElems);
    __aicore__ inline void Stage1Sort(uint32_t totalElems);
    __aicore__ inline void ScatterOutputs(uint32_t totalElems);
    __aicore__ inline void BuildStage2Keys(uint32_t totalElems);
    __aicore__ inline void Stage2Sort(uint32_t totalElems);
    __aicore__ inline void BuildOutputs(uint32_t totalElems);
    __aicore__ inline void StoreBatch(int64_t segStart, uint32_t totalElems);

private:
    // Two-stage sort strategy for small-axis sorting:
    // Stage 1: Sort all segments as one flattened array (value + original flat index).
    // Stage 2: Restore per-segment ordering via packed keys (segId*totalElems+rank) to regroup by segment.
    static constexpr SortConfig kStage1SortConfig{ SortType::RADIX_SORT, IsDescend };
    static constexpr SortConfig kStage2SortConfig{ SortType::RADIX_SORT, false };

    TPipe *pipe_ = nullptr;
    const SortRegBaseTilingData *tilingData_ = nullptr;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<OutIdxT> outIdxGm_;

    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> stage1ValueBuf_;
    TBuf<TPosition::VECCALC> stage1OrderBuf_;
    TBuf<TPosition::VECCALC> finalIdxBuf_;
    TBuf<TPosition::VECCALC> rankInverseBuf_;
    TBuf<TPosition::VECCALC> stage2OrderBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    LocalTensor<T> inputValues_;
    LocalTensor<T> stage1Values_;
    LocalTensor<uint32_t> stage1Order_;
    LocalTensor<uint32_t> rankInverse_;
    LocalTensor<uint32_t> stage2KeysIn_;
    LocalTensor<uint32_t> stage2KeysOut_;
    LocalTensor<uint32_t> stage2Order_;
    LocalTensor<T> finalValues_;
    LocalTensor<OutIdxT> finalIdx_;
    LocalTensor<uint8_t> tmp_;

    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t batchNum_ = 0;
    uint32_t segmentLen_ = 0;
    uint32_t maxFlatElems_ = 0;
    uint32_t blockUbSize_ = 0;
    int64_t totalSegs_ = 0;
    bool useRankInverse_ = false;
};

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline uint32_t SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::AlignUpBytes(uint32_t bytes) const
{
    if (blockUbSize_ == 0) {
        return bytes;
    }
    return ((bytes + blockUbSize_ - 1U) / blockUbSize_) * blockUbSize_;
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::Init(GM_ADDR x, GM_ADDR y,
    GM_ADDR idx, GM_ADDR workspace, const SortRegBaseTilingData *tilingData, TPipe *pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    pipe_ = pipe;
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    batchSize_ = tilingData_->keyParams0;
    batchNum_ = tilingData_->keyParams1;
    segmentLen_ = tilingData_->numTileDataSize;
    maxFlatElems_ = batchSize_ * segmentLen_;
    blockUbSize_ = Ops::Base::GetUbBlockSize();
    totalSegs_ = tilingData_->unsortedDimNum;
    useRankInverse_ = tilingData_->keyParams2 != 0U;

    inputXGm_.SetGlobalBuffer((__gm__ T *)x);
    outValueGm_.SetGlobalBuffer((__gm__ T *)y);
    outIdxGm_.SetGlobalBuffer((__gm__ OutIdxT *)idx);

    if (batchSize_ == 0 || segmentLen_ == 0 || maxFlatElems_ == 0) {
        return;
    }

    // Each buffer stores one flattened batch. Rows are UB-block aligned to match host sizing.
    pipe_->InitBuffer(inputBuf_, AlignUpBytes(maxFlatElems_ * sizeof(T)));
    pipe_->InitBuffer(stage1ValueBuf_, AlignUpBytes(maxFlatElems_ * sizeof(T)));
    pipe_->InitBuffer(stage1OrderBuf_, AlignUpBytes(maxFlatElems_ * sizeof(uint32_t)));
    constexpr uint32_t kAliasElemBytes =
        (sizeof(OutIdxT) > sizeof(uint32_t)) ? static_cast<uint32_t>(sizeof(OutIdxT)) :
                                               static_cast<uint32_t>(sizeof(uint32_t));
    pipe_->InitBuffer(finalIdxBuf_, AlignUpBytes(maxFlatElems_ * kAliasElemBytes));
    pipe_->InitBuffer(rankInverseBuf_, AlignUpBytes(maxFlatElems_ * sizeof(uint32_t)));
    if (!useRankInverse_) {
        pipe_->InitBuffer(stage2OrderBuf_, AlignUpBytes(maxFlatElems_ * sizeof(uint32_t)));
    }
    pipe_->InitBuffer(tmpBuf_, tilingData_->tmpUbSize);

    inputValues_ = inputBuf_.Get<T>();
    stage1Values_ = stage1ValueBuf_.Get<T>();
    stage1Order_ = stage1OrderBuf_.Get<uint32_t>();
    // finalValues_ intentionally aliases inputValues_. After Stage1Sort finishes,
    // inputValues_ is no longer read in the same batch, so the input buffer can be
    // reused as the final output buffer without overlap.
    finalValues_ = inputValues_;
    finalIdx_ = finalIdxBuf_.Get<OutIdxT>();
    if (useRankInverse_) {
        rankInverse_ = rankInverseBuf_.Get<uint32_t>();
    } else {
        // The two-stage-sort path reuses finalIdxBuf_ as stage2KeysIn_ before final indices are built.
        stage2KeysIn_ = finalIdxBuf_.Get<uint32_t>();
        stage2KeysOut_ = rankInverseBuf_.Get<uint32_t>();
        stage2Order_ = stage2OrderBuf_.Get<uint32_t>();
    }
    tmp_ = tmpBuf_.Get<uint8_t>();
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline uint32_t SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::ComputeValidSegs(uint32_t batchId) const
{
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    int64_t segRemain = totalSegs_ - segStart;
    if (segRemain <= 0) {
        return 0;
    }
    if (segRemain >= static_cast<int64_t>(batchSize_)) {
        return batchSize_;
    }
    return static_cast<uint32_t>(segRemain);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::LoadBatch(int64_t segStart,
    uint32_t totalElems)
{
    DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
    DataCopyExtParams copyParam{ 1, static_cast<uint32_t>(totalElems * sizeof(T)), 0, 0, 0 };
    int64_t gmOffset = segStart * static_cast<int64_t>(segmentLen_);
    DataCopyPad(inputValues_, inputXGm_[gmOffset], copyParam, padParams);
    // MTE2 writes inputValues_; the vector Sort consumes it.
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::Stage1Sort(uint32_t totalElems)
{
    AscendC::Sort<T, false, kStage1SortConfig>(stage1Values_, stage1Order_, inputValues_, tmp_, totalElems);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::ScatterOutputs(uint32_t totalElems)
{
    // Stage 1 Sort writes stage1Values_/stage1Order_; VF consumes them.
    asc_vf_call<RankInverseScatter<T, OutIdxT>>(dim3(TWO_STAGE_THREAD_NUM),
        totalElems, segmentLen_,
        (__ubuf__ T *)stage1Values_.GetPhyAddr(),
        (__ubuf__ uint32_t *)stage1Order_.GetPhyAddr(),
        (__ubuf__ uint32_t *)rankInverse_.GetPhyAddr(),
        (__ubuf__ T *)finalValues_.GetPhyAddr(),
        (__ubuf__ OutIdxT *)finalIdx_.GetPhyAddr());
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::BuildStage2Keys(
    uint32_t totalElems)
{
    // Stage 1 Sort writes stage1Order_; VF consumes it.
    asc_vf_call<BuildStage2KeysSimt<T, OutIdxT>>(dim3(TWO_STAGE_THREAD_NUM),
        totalElems, segmentLen_,
        (__ubuf__ uint32_t *)stage1Order_.GetPhyAddr(),
        (__ubuf__ uint32_t *)stage2KeysIn_.GetPhyAddr());
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::Stage2Sort(uint32_t totalElems)
{
    AscendC::Sort<uint32_t, false, kStage2SortConfig>(stage2KeysOut_, stage2Order_, stage2KeysIn_, tmp_, totalElems);
}

// Reconstruct final outputs using stage-2 order to gather from stage-1 results.
// Full pipeline example (2x3, ascend): Input [[3,1,5], [4,2,6]] -> flatten [3,1,5,4,2,6]
//   Stage1 sort: stage1Values=[1,2,3,4,5,6], stage1Order=[1,4,0,3,2,5] (flatIdx per rank)
//   BuildStage2Keys: stage2Keys=[0,7,2,9,4,11] (key=segId*6+rank), stage1Order->laneIds [1,1,0,0,2,2]
//   Stage2 sort: stage2KeysOut=[0,2,4,7,9,11], stage2Order=[0,2,4,1,3,5] (final permutation)
//   BuildOutputs: gather stage1Values/Order via stage2Order -> values=[1,3,5,2,4,6], indices=[1,0,2,1,0,2]
template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::BuildOutputs(uint32_t totalElems)
{
    LocalTensor<uint32_t> gatherOffsets = stage2KeysOut_.ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> gatherOffsetsInt = stage2KeysOut_.ReinterpretCast<int32_t>();

    Muls(gatherOffsetsInt, stage2Order_.ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(T)), totalElems);
    Gather(finalValues_, stage1Values_, gatherOffsets, 0, totalElems);

    Muls(gatherOffsetsInt, stage2Order_.ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(uint32_t)), totalElems);
    if constexpr (IsSameType<OutIdxT, int64_t>::value) {
        LocalTensor<int32_t> gatheredIdxInt32 = stage2Order_.ReinterpretCast<int32_t>();
        Gather(gatheredIdxInt32, stage1Order_.ReinterpretCast<int32_t>(), gatherOffsets, 0, totalElems);
        Cast(finalIdx_, gatheredIdxInt32, RoundMode::CAST_NONE, Ops::Base::CeilAlign(totalElems, 4u));
    } else {
        Gather(finalIdx_.template ReinterpretCast<int32_t>(), stage1Order_.ReinterpretCast<int32_t>(),
            gatherOffsets, 0, totalElems);
    }
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::StoreBatch(int64_t segStart,
    uint32_t totalElems)
{
    int64_t gmOffset = segStart * static_cast<int64_t>(segmentLen_);
    DataCopyExtParams valueCopyParam{ 1, static_cast<uint32_t>(totalElems * sizeof(T)), 0, 0, 0 };
    DataCopyExtParams idxCopyParam{ 1, static_cast<uint32_t>(totalElems * sizeof(OutIdxT)), 0, 0, 0 };
    // Rank-inverse writes final buffers via SIMT VF; BuildOutputs writes them via vector APIs.
    // Wait for the producing VF/vector work before GM writeback.
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(outValueGm_[gmOffset], finalValues_, valueCopyParam);
    DataCopyPad(outIdxGm_[gmOffset], finalIdx_, idxCopyParam);

    // finalValues_ aliases inputValues_; next LoadBatch uses MTE2 to overwrite that UB.
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    // The next loop iteration is issued by scalar/control flow, so wait on MTE3_S, not MTE3_V.
    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::Process()
{
    if (blockIdx_ >= blockDim_ || batchSize_ == 0 || segmentLen_ == 0) {
        return;
    }

    // Neighboring cores process neighboring batches to preserve sequential GM access.
    uint32_t batchesPerCore = (batchNum_ + blockDim_ - 1) / blockDim_;
    uint32_t startBatch = blockIdx_ * batchesPerCore;
    uint32_t endBatch = (startBatch + batchesPerCore < batchNum_) ? (startBatch + batchesPerCore) : batchNum_;
    for (uint32_t batchId = startBatch; batchId < endBatch; ++batchId) {
        uint32_t validSegs = ComputeValidSegs(batchId);
        if (validSegs == 0) {
            continue;
        }
        uint32_t totalElems = validSegs * segmentLen_;
        int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
        // Pipeline: load -> value sort -> restore per-segment order -> store.
        LoadBatch(segStart, totalElems);
        Stage1Sort(totalElems);
        if (useRankInverse_) {
            ScatterOutputs(totalElems);
        } else {
            BuildStage2Keys(totalElems);
            Stage2Sort(totalElems);
            BuildOutputs(totalElems);
        }
        StoreBatch(segStart, totalElems);
    }
}

} // namespace Sort

#endif
