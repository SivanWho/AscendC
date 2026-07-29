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
 * \file clip_by_value_v2_ass.h
 * \brief
 */
#ifndef CLIP_BY_VALUE_V2_ASS_H
#define CLIP_BY_VALUE_V2_ASS_H

#include "kernel_operator.h"
namespace ClipByValueV2 {

using namespace AscendC;

template <typename T>
class ClipByValueV2Ass {
public:
    __aicore__ inline ClipByValueV2Ass(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace,
        const ClipByValueSigDimTilingData* tilingData, TPipe* pipePtr);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum = 2;

protected:
    __aicore__ inline void CopyIn(const int64_t& gmOffset);
    __aicore__ inline void Compute(const int64_t& len);
    __aicore__ inline void CopyOut(const int64_t& gmOffset);

protected:
    int64_t blockIdx{0};
    int64_t blockNum{0};
    int64_t blockLoopNum{0};
    int64_t blockLoopNumWithoutTail{0};
    int64_t ubFormerNum{0};
    int64_t ubTailNum{0};
    int64_t xDim{0};
    T minValue{0};
    T maxValue{0};

    TPipe* pipe{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueue;
    GlobalTensor<T> xGm, minGm, maxGm, yGm;

    DataCopyExtParams dataCopyParams;
    DataCopyPadExtParams<T> padParams;
};

template <typename T>
__aicore__ inline void ClipByValueV2Ass<T>::Init(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace, const ClipByValueSigDimTilingData* tilingData,
    TPipe* pipePtr)
{
    pipe = pipePtr;
    blockIdx = GetBlockIdx();
    blockNum = tilingData->blockNum;
    if (blockIdx >= blockNum) {
        return;
    }

    ubFormerNum = tilingData->ubFormer;
    if (blockIdx == (blockNum - 1)) {
        // 尾核，或者无尾核场景下的最后一个核.需要在该核处理ub的尾块
        blockLoopNum = tilingData->blockTail;
        ubTailNum = tilingData->ubTail;
    } else {
        blockLoopNum = tilingData->blockFormer;
        ubTailNum = ubFormerNum; // 非最后一个核，不需要处理尾块，都是整块ub
    }

    blockLoopNumWithoutTail = blockLoopNum - 1;
    xDim = tilingData->xDim;

    int64_t gmBlockOffset = blockIdx * tilingData->blockFormer * tilingData->ubFormer;
    xGm.SetGlobalBuffer((__gm__ T*)x + gmBlockOffset);
    minGm.SetGlobalBuffer((__gm__ T*)min);
    maxGm.SetGlobalBuffer((__gm__ T*)max);
    yGm.SetGlobalBuffer((__gm__ T*)y + gmBlockOffset);

    minValue = minGm.GetValue(0);
    maxValue = maxGm.GetValue(0);

    pipe->InitBuffer(inQueueX, bufferNum, tilingData->ubFormer * sizeof(T));
    pipe->InitBuffer(outQueue, bufferNum, tilingData->ubFormer * sizeof(T));

    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 0;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
}

template <typename T>
__aicore__ inline void ClipByValueV2Ass<T>::CopyIn(const int64_t& gmOffset)
{
    auto xLocalIn = inQueueX.template AllocTensor<T>();
    DataCopyPad(xLocalIn, xGm[gmOffset], dataCopyParams, padParams);
    inQueueX.EnQue(xLocalIn);
}

template <typename T>
__aicore__ inline void ClipByValueV2Ass<T>::Compute(const int64_t& len)
{
    auto xLocal = inQueueX.template DeQue<T>();
    auto yLocal = outQueue.template AllocTensor<T>();
    Maxs(yLocal, xLocal, minValue, len);
    inQueueX.FreeTensor(xLocal);
    Mins(yLocal, yLocal, maxValue, len);
    outQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void ClipByValueV2Ass<T>::CopyOut(const int64_t& gmOffset)
{
    auto yLocalOut = outQueue.template DeQue<T>();
    DataCopyPad(yGm[gmOffset], yLocalOut, dataCopyParams);
    outQueue.FreeTensor(yLocalOut);
}

template <typename T>
__aicore__ inline void ClipByValueV2Ass<T>::Process()
{
    if (blockIdx >= blockNum) {
        return;
    }

    // tiling 按照128对齐，必然是整32B block的
    dataCopyParams.blockLen = ubFormerNum * sizeof(T);
    int64_t gmOffset = 0;
    for (int64_t i = 0; i < blockLoopNumWithoutTail; ++i) {
        CopyIn(gmOffset);
        Compute(ubFormerNum);
        CopyOut(gmOffset);
        gmOffset += ubFormerNum;
    }

    // 处理最后一次，可能是ub 尾块, params共用，可以接受
    dataCopyParams.blockLen = ubTailNum * sizeof(T);
    CopyIn(gmOffset);
    Compute(ubTailNum);
    CopyOut(gmOffset);
}

} // namespace ClipByValueV2
#endif // CLIP_BY_VALUE_V2_ASS_H