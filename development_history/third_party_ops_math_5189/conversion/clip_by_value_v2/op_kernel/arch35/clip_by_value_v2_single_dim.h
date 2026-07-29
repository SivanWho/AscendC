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
 * \file clip_by_value_v2_single_dim.h
 * \brief
 */
#ifndef CLIP_BY_VALUE_V2_SINGLE_DIM_H
#define CLIP_BY_VALUE_V2_SINGLE_DIM_H

#include "kernel_operator.h"
namespace ClipByValueV2 {
using namespace AscendC;

template <typename T>
class ClipByValueV2SigDim {
public:
    __aicore__ inline ClipByValueV2SigDim(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace,
        const ClipByValueSigDimTilingData* tilingData, TPipe* pipePtr);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum = 2;

protected:
    __aicore__ inline void CommonInit(GM_ADDR x, GM_ADDR y, const ClipByValueSigDimTilingData* tilingData);
    __aicore__ inline void PrepareCacheSingleDim();
    __aicore__ inline void CopyIn(const int64_t& gmOffset);
    __aicore__ inline void Compute(const int64_t& len);
    __aicore__ inline void CopyOut(const int64_t& gmOffset);
    __aicore__ inline void PostCacheSingleDim();

protected:
    int64_t blockIdx{0};
    int64_t blockNum{0};
    int64_t blockLoopNum{0};
    int64_t blockLoopNumWithoutTail{0};
    int64_t ubFormerNum{0};
    int64_t ubTailNum{0};
    int64_t xDim{0};
    int64_t minDim{0};
    int64_t maxDim{0};
    int64_t gmBlockOffset{0};

    TPipe* pipe{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueMin;
    TQue<QuePosition::VECIN, 1> inQueueMax;
    TQue<QuePosition::VECOUT, 1> outQueue;
    GlobalTensor<T> xGm, minGm, maxGm, yGm;
    LocalTensor<T> xLocal, minLocal, maxLocal;

    DataCopyExtParams dataCopyParams;
    DataCopyPadExtParams<T> padParams;
};

template <typename T>
__aicore__ inline void ClipByValueV2SigDim<T>::Init(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace, const ClipByValueSigDimTilingData* tilingData,
    TPipe* pipePtr)
{
    pipe = pipePtr;
    blockIdx = GetBlockIdx();
    blockNum = tilingData->blockNum;
    if (blockIdx >= blockNum) {
        return;
    }

    CommonInit(x, y, tilingData);

    minDim = tilingData->minDim;
    maxDim = tilingData->maxDim;

    if (minDim == 1) {
        minGm.SetGlobalBuffer((__gm__ T*)min);
    } else {
        minGm.SetGlobalBuffer((__gm__ T*)min + gmBlockOffset);
    }

    if (maxDim == 1) {
        maxGm.SetGlobalBuffer((__gm__ T*)max);
    } else {
        maxGm.SetGlobalBuffer((__gm__ T*)max + gmBlockOffset);
    }
}

template <typename T>
__aicore__ inline void ClipByValueV2SigDim<T>::CommonInit(
    GM_ADDR x, GM_ADDR y, const ClipByValueSigDimTilingData* tilingData)
{
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

    gmBlockOffset = blockIdx * tilingData->blockFormer * tilingData->ubFormer;
    if (xDim == 1) {
        xGm.SetGlobalBuffer((__gm__ T*)x);
    } else {
        xGm.SetGlobalBuffer((__gm__ T*)x + gmBlockOffset);
    }

    yGm.SetGlobalBuffer((__gm__ T*)y + gmBlockOffset);

    pipe->InitBuffer(inQueueX, bufferNum, tilingData->ubFormer * sizeof(T));
    pipe->InitBuffer(inQueueMin, bufferNum, tilingData->ubFormer * sizeof(T));
    pipe->InitBuffer(inQueueMax, bufferNum, tilingData->ubFormer * sizeof(T));
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
__aicore__ inline void ClipByValueV2SigDim<T>::PrepareCacheSingleDim()
{
    if (xDim == 1) {
        T xScalar = xGm.GetValue(0);
        xLocal = inQueueX.template AllocTensor<T>();
        Duplicate(xLocal, xScalar, ubFormerNum);
    }

    if (minDim == 1) {
        T minScalar = minGm.GetValue(0);
        minLocal = inQueueMin.template AllocTensor<T>();
        Duplicate(minLocal, minScalar, ubFormerNum);
    }

    if (maxDim == 1) {
        T maxScalar = maxGm.GetValue(0);
        maxLocal = inQueueMax.template AllocTensor<T>();
        Duplicate(maxLocal, maxScalar, ubFormerNum);
    }
}

template <typename T>
__aicore__ inline void ClipByValueV2SigDim<T>::CopyIn(const int64_t& gmOffset)
{
    // 读取输入时候用新的LocalTensor，不复用xLocal等，增加并行度
    if (xDim != 1) {
        auto xLocalIn = inQueueX.template AllocTensor<T>();
        DataCopyPad(xLocalIn, xGm[gmOffset], dataCopyParams, padParams);
        inQueueX.EnQue(xLocalIn);
    }

    if (minDim != 1) {
        auto minLocalIn = inQueueMin.template AllocTensor<T>();
        DataCopyPad(minLocalIn, minGm[gmOffset], dataCopyParams, padParams);
        inQueueMin.EnQue(minLocalIn);
    }

    if (maxDim != 1) {
        auto maxLocalIn = inQueueMax.template AllocTensor<T>();
        DataCopyPad(maxLocalIn, maxGm[gmOffset], dataCopyParams, padParams);
        inQueueMax.EnQue(maxLocalIn);
    }
}

template <typename T>
__aicore__ inline void ClipByValueV2SigDim<T>::Compute(const int64_t& len)
{
    if (xDim != 1) {
        xLocal = inQueueX.template DeQue<T>();
    }
    if (minDim != 1) {
        minLocal = inQueueMin.template DeQue<T>();
    }
    auto yLocal = outQueue.template AllocTensor<T>();
    Max(yLocal, xLocal, minLocal, len);
    if (xDim != 1) {
        inQueueX.FreeTensor(xLocal);
    }
    if (minDim != 1) {
        inQueueMin.FreeTensor(minLocal);
    }

    if (maxDim != 1) {
        maxLocal = inQueueMax.template DeQue<T>();
    }
    Min(yLocal, yLocal, maxLocal, len);
    if (maxDim != 1) {
        inQueueMax.FreeTensor(maxLocal);
    }

    outQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void ClipByValueV2SigDim<T>::CopyOut(const int64_t& gmOffset)
{
    auto yLocalOut = outQueue.template DeQue<T>();
    DataCopyPad(yGm[gmOffset], yLocalOut, dataCopyParams);
    outQueue.FreeTensor(yLocalOut);
}

template <typename T>
__aicore__ inline void ClipByValueV2SigDim<T>::PostCacheSingleDim()
{
    if (xDim == 1) {
        inQueueX.FreeTensor(xLocal);
    }

    if (minDim == 1) {
        inQueueMin.FreeTensor(minLocal);
    }

    if (maxDim == 1) {
        inQueueMax.FreeTensor(maxLocal);
    }
}

template <typename T>
__aicore__ inline void ClipByValueV2SigDim<T>::Process()
{
    if (blockIdx >= blockNum) {
        return;
    }

    PrepareCacheSingleDim();

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

    // 释放缓存等资源
    PostCacheSingleDim();
}

} // namespace ClipByValueV2
#endif // CLIP_BY_VALUE_V2_SINGLE_DIM_H