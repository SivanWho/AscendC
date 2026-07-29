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
 * \file clip_by_value_v2_s64_nddma_with_loops.h
 * \brief
 */

#ifndef ASCENDC_CLIP_BY_VALUE_V2_S64_NDDMA_WITH_LOOPS_H_
#define ASCENDC_CLIP_BY_VALUE_V2_S64_NDDMA_WITH_LOOPS_H_

#include "kernel_operator.h"
#include "atvoss/util/broadcast_utils.h"

namespace ClipByValueV2 {

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

class ClipByValueV2S64NddmaWithLoops {
public:
    __aicore__ inline ClipByValueV2S64NddmaWithLoops(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR clipValueMin, GM_ADDR clipValueMax, GM_ADDR y, GM_ADDR workspace,
        const ClipByValueTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmX_.SetGlobalBuffer((__gm__ int64_t*)x);
        inputGmClipValueMin_.SetGlobalBuffer((__gm__ int64_t*)clipValueMin);
        inputGmClipValueMax_.SetGlobalBuffer((__gm__ int64_t*)clipValueMax);
        outputGmY_.SetGlobalBuffer((__gm__ int64_t*)y);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(int64_t);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn2_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->blockTail :
                                                                                   tilingDataPtr_->blockFormer;
        int64_t axesIndices[Ops::Base::BROADCAST_MAX_DIMS] = {0};
        Ops::Base::BroadcastGetAxesIndices(
            axesIndices, tilingDataPtr_->blockFormer * AscendC::GetBlockIdx(), tilingDataPtr_->outputDims,
            tilingDataPtr_->ubSplitAxis, tilingDataPtr_->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            if (ubLoopIdx != 0) {
                Ops::Base::BroadcastUpdateAxesIndices(
                    axesIndices, tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubOuter);
            }
            int64_t ubSplitSize = axesIndices[tilingDataPtr_->ubSplitAxis] == tilingDataPtr_->ubOuter - 1 ?
                                      tilingDataPtr_->ubTail :
                                      tilingDataPtr_->ubFormer;
            CopyIn0(ubSplitSize, axesIndices, ubLoopIdx);
            CopyIn1(ubSplitSize, axesIndices, ubLoopIdx);
            CopyIn2(ubSplitSize, axesIndices, ubLoopIdx);
            Compute3(ubSplitSize, axesIndices);
            CopyOut4(ubSplitSize, axesIndices, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(
        int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<int64_t>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithLoop(
                inputGmX_, bufferIn0_, tilingDataPtr_->outputDims, tilingDataPtr_->outputStrides,
                tilingDataPtr_->input0Strides, axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn0_.EnQue<int64_t>(bufferIn0_);
    }

    __aicore__ inline void CopyIn1(
        int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<int64_t>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithLoop(
                inputGmClipValueMin_, bufferIn1_, tilingDataPtr_->outputDims, tilingDataPtr_->outputStrides,
                tilingDataPtr_->input1Strides, axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn1_.EnQue<int64_t>(bufferIn1_);
    }

    __aicore__ inline void CopyIn2(
        int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn2_ = queIn2_.AllocTensor<int64_t>();
        if ((tilingDataPtr_->input2Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithLoop(
                inputGmClipValueMax_, bufferIn2_, tilingDataPtr_->outputDims, tilingDataPtr_->outputStrides,
                tilingDataPtr_->input2Strides, axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn2_.EnQue<int64_t>(bufferIn2_);
    }

    __aicore__ inline void Compute3(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS])
    {
        bufferIn0_ = queIn0_.DeQue<int64_t>();
        bufferIn1_ = queIn1_.DeQue<int64_t>();
        bufferIn2_ = queIn2_.DeQue<int64_t>();
        bufferOut0_ = queOut0_.AllocTensor<int64_t>();
        Max(bufferOut0_, bufferIn0_, bufferIn1_,
            ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis]);
        Min(bufferOut0_, bufferOut0_, bufferIn2_,
            ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis]);
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queIn2_.FreeTensor(bufferIn2_);
        queOut0_.EnQue<int64_t>(bufferOut0_);
    }

    __aicore__ inline void CopyOut4(
        int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<int64_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen =
            ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] * sizeof(int64_t);
        int64_t gmOffset = Ops::Base::BroadcastGetGmOffset(
            axesIndices, tilingDataPtr_->outputStrides, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmY_[gmOffset], bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const ClipByValueTilingData* tilingDataPtr_;
    GlobalTensor<int64_t> inputGmX_;
    GlobalTensor<int64_t> inputGmClipValueMin_;
    GlobalTensor<int64_t> inputGmClipValueMax_;
    GlobalTensor<int64_t> outputGmY_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn2_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<int64_t> bufferIn0_;
    LocalTensor<int64_t> bufferIn1_;
    LocalTensor<int64_t> bufferIn2_;
    LocalTensor<int64_t> bufferOut0_;
};

} // namespace ClipByValueV2
#endif // ASCENDC_CLIP_BY_VALUE_V2_S64_NDDMA_WITH_LOOPS_H_