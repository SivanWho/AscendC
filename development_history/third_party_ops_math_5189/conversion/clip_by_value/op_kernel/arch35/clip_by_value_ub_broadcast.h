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
 * \file clip_by_value_ub_broadcast.h
 * \brief
 */

#ifndef ASCENDC_CLIP_BY_VALUEUB_BROADCAST_H_
#define ASCENDC_CLIP_BY_VALUEUB_BROADCAST_H_

#include "kernel_operator.h"
#include "atvoss/util/broadcast_utils.h"
using namespace Ops::Base;
namespace ClipByValue {

using AscendC::BroadcastTiling;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

template <typename T, int64_t R>
class ClipByValueUbBroadcast {
public:
    __aicore__ inline ClipByValueUbBroadcast(){};

    __aicore__ inline void Process();
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR clipValueMin, GM_ADDR clipValueMax, GM_ADDR y, GM_ADDR workspace,
        const ClipByValueUbBrcTilingData* tilingDataPtr, TPipe* pipePtr);

private:
    __aicore__ inline void CopyIn0(
        int64_t ubSplitSize, const int64_t (&axesIndices)[8], int64_t ubLoopIdx, uint32_t (&srcShape)[8],
        uint32_t (&dstShape)[8], int64_t input_ub_length, BroadcastTiling& broadcastTiling);
    __aicore__ inline void CopyIn1(
        int64_t ubSplitSize, const int64_t (&axesIndices)[8], int64_t ubLoopIdx, uint32_t (&srcShape)[8],
        uint32_t (&dstShape)[8], int64_t input_ub_length, BroadcastTiling& broadcastTiling);
    __aicore__ inline void CopyIn2(
        int64_t ubSplitSize, const int64_t (&axesIndices)[8], int64_t ubLoopIdx, uint32_t (&srcShape)[8],
        uint32_t (&dstShape)[8], int64_t input_ub_length, BroadcastTiling& broadcastTiling);
    __aicore__ inline void Compute3(int64_t ubSplitSize, const int64_t (&axesIndices)[8]);
    __aicore__ inline void CopyOut4(int64_t ubSplitSize, const int64_t (&axesIndices)[8], int64_t ubLoopIdx);

private:
    TPipe* pipePtr_;
    const ClipByValueUbBrcTilingData* tilingDataPtr_;
    GlobalTensor<T> inputGmX_;
    GlobalTensor<T> inputGmClipValueMin_;
    GlobalTensor<T> inputGmClipValueMax_;
    GlobalTensor<T> outputGmY_;
    TQue<QuePosition::VECIN, 1> queIn0_;
    TQue<QuePosition::VECIN, 1> queIn1_;
    TQue<QuePosition::VECIN, 1> queIn2_;
    TQue<QuePosition::VECOUT, 1> queOut0_;

    TBuf<QuePosition::VECCALC> broadcastTempBuf0;
    TBuf<QuePosition::VECCALC> broadcastTempBuf1;
    TBuf<QuePosition::VECCALC> broadcastTempBuf2;

    int64_t input0Ublength[2] = {1, 1};
    int64_t input1Ublength[2] = {1, 1};
    int64_t input2Ublength[2] = {1, 1};
    int64_t dstUblength[2] = {1, 1};

    uint32_t input0UbFormershape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t input1UbFormershape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t input2UbFormershape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t dstUbFormershape[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    uint32_t input0UbTailshape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t input1UbTailshape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t input2UbTailshape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t dstUbTailshape[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    BroadcastTiling input0FormerBroadcastTiling;
    BroadcastTiling input1FormerBroadcastTiling;
    BroadcastTiling input2FormerBroadcastTiling;

    BroadcastTiling input0TailBroadcastTiling;
    BroadcastTiling input1TailBroadcastTiling;
    BroadcastTiling input2TailBroadcastTiling;
};

template <size_t N>
__aicore__ inline void GetUbBroadcastShapeInfo(
    const int64_t (&oriShape)[N], int64_t inputUblength[2], uint32_t (&ubFormershape)[N], uint32_t (&ubTailshape)[N],
    const ClipByValueUbBrcTilingData* tilingDataPtr_)
{
    int64_t ubFormerLength = 1;
    int64_t ubTailLength = 1;
    if (oriShape[tilingDataPtr_->ubSplitAxis] == 1) {
        ubFormershape[0] = 1;
    } else {
        ubTailshape[0] = tilingDataPtr_->ubTail;
        ubTailLength = tilingDataPtr_->ubTail;
        ubFormershape[0] = tilingDataPtr_->ubFormer;
        ubFormerLength = tilingDataPtr_->ubFormer;
    }

    int j = 1;
    for (uint64_t i = tilingDataPtr_->ubSplitAxis + 1; i < tilingDataPtr_->shapeLen; i++) {
        ubFormershape[j] = oriShape[i];
        ubTailshape[j] = oriShape[i];
        ubFormerLength *= oriShape[i];
        ubTailLength *= oriShape[i];
        j = j + 1;
    }

    inputUblength[0] = ubFormerLength;
    inputUblength[1] = ubTailLength;
}

template <typename T1>
__aicore__ inline void CopyInDataCopyPad(
    LocalTensor<T1>& inputUb, AscendC::GlobalTensor<T1>& inputGm, const int64_t& gmOffset, const int64_t& dataLength)
{
    AscendC::DataCopyExtParams dataCopyExtParams;
    AscendC::DataCopyPadExtParams<T1> dataCopyPadExtParams;
    dataCopyExtParams.blockCount = 1;
    dataCopyExtParams.blockLen = dataLength * sizeof(T1);
    AscendC::DataCopyPad(inputUb, inputGm[gmOffset], dataCopyExtParams, dataCopyPadExtParams);
}

template <typename T1, int64_t R1, size_t N>
__aicore__ inline void CopyInAndBroadcast(
    AscendC::GlobalTensor<T1>& inputGm, LocalTensor<T1>& broadcastTempBuf, TQue<QuePosition::VECIN, 1>& inputQueue,
    const int64_t (&inputStrides)[N], const int64_t (&outputStrides)[N], uint32_t (&dstShape)[N],
    uint32_t (&srcShape)[N], const int64_t (&axesIndices)[N], int64_t ubSplitAxis, int64_t input_ub_length,
    int64_t ubSplitSize, int64_t ubFormer, BroadcastTiling& broadcastTiling)
{
    // 根据复原的索引信息，计算UB内broadcast的偏移地址
    int64_t gmOffset = BroadcastGetGmOffset(axesIndices, inputStrides, ubSplitAxis, ubFormer);

    LocalTensor<T1> inputBuffer = inputQueue.AllocTensor<T1>();
    // 如果UB切分轴以内的输入输出轴大小相等，则直接用move align.
    if (outputStrides[ubSplitAxis] != inputStrides[ubSplitAxis]) {
        // 走ub内broadcast逻辑，分两步
        // 1、将小shape拷贝到UB内
        CopyInDataCopyPad(inputBuffer, inputGm, gmOffset, input_ub_length);

        // 2、设置同步MTE2->Vector
        inputQueue.EnQue(inputBuffer);
        inputBuffer = inputQueue.DeQue<T1>();

        // 只处理fp16、bf16和tiling对应，接口暂不支持int64等类型
        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            if constexpr (R1 == -1) {
                Broadcast(broadcastTempBuf, inputBuffer, dstShape, srcShape, &broadcastTiling);
            } else {
                // 3、对小shape做UB内broadcast逻辑处理
                Broadcast<T1, R1>(broadcastTempBuf, inputBuffer, dstShape, srcShape, &broadcastTiling);
            }
        }
    } else {
        // 走move align逻辑
        CopyInDataCopyPad(inputBuffer, inputGm, gmOffset, input_ub_length);
        // 2、设置同步操作MTE2->Vector
        inputQueue.EnQue(inputBuffer);
        inputBuffer = inputQueue.DeQue<T1>();
        Muls(broadcastTempBuf, inputBuffer, (T1)1.0, input_ub_length);
    }
    inputQueue.FreeTensor(inputBuffer);
}

template <typename T, int64_t R>
__aicore__ inline void ClipByValueUbBroadcast<T, R>::Init(
    GM_ADDR x, GM_ADDR clipValueMin, GM_ADDR clipValueMax, GM_ADDR y, GM_ADDR workspace,
    const ClipByValueUbBrcTilingData* tilingDataPtr, TPipe* pipePtr)
{
    pipePtr_ = pipePtr;
    tilingDataPtr_ = tilingDataPtr;
    inputGmX_.SetGlobalBuffer((__gm__ T*)x);
    inputGmClipValueMin_.SetGlobalBuffer((__gm__ T*)clipValueMin);
    inputGmClipValueMax_.SetGlobalBuffer((__gm__ T*)clipValueMax);
    outputGmY_.SetGlobalBuffer((__gm__ T*)y);
    constexpr int64_t DOUBLE_BUFFER = 2;
    pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, tilingDataPtr_->buffSize * sizeof(T));
    pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, tilingDataPtr_->buffSize * sizeof(T));
    pipePtr_->InitBuffer(queIn2_, DOUBLE_BUFFER, tilingDataPtr_->buffSize * sizeof(T));
    pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, tilingDataPtr_->buffSize * sizeof(T));

    pipePtr_->InitBuffer(broadcastTempBuf0, tilingDataPtr_->buffSize * sizeof(T));
    pipePtr_->InitBuffer(broadcastTempBuf1, tilingDataPtr_->buffSize * sizeof(T));
    pipePtr_->InitBuffer(broadcastTempBuf2, tilingDataPtr_->buffSize * sizeof(T));

    // 根据原始shape 获取
    GetUbBroadcastShapeInfo(
        tilingDataPtr->input0Dims, input0Ublength, input0UbFormershape, input0UbTailshape, tilingDataPtr_);
    GetUbBroadcastShapeInfo(
        tilingDataPtr->input1Dims, input1Ublength, input1UbFormershape, input1UbTailshape, tilingDataPtr_);
    GetUbBroadcastShapeInfo(
        tilingDataPtr->input2Dims, input2Ublength, input2UbFormershape, input2UbTailshape, tilingDataPtr_);
    GetUbBroadcastShapeInfo(tilingDataPtr->outputDims, dstUblength, dstUbFormershape, dstUbTailshape, tilingDataPtr_);

    if constexpr (R == -1) {
        GetBroadcastTilingInfo<T>(
            tilingDataPtr->runningRank, dstUbFormershape, input0UbFormershape, false, input0FormerBroadcastTiling);
        GetBroadcastTilingInfo<T>(
            tilingDataPtr->runningRank, dstUbFormershape, input1UbFormershape, false, input1FormerBroadcastTiling);
        GetBroadcastTilingInfo<T>(
            tilingDataPtr->runningRank, dstUbFormershape, input2UbFormershape, false, input2FormerBroadcastTiling);
        GetBroadcastTilingInfo<T>(
            tilingDataPtr->runningRank, dstUbFormershape, input0UbTailshape, false, input0TailBroadcastTiling);
        GetBroadcastTilingInfo<T>(
            tilingDataPtr->runningRank, dstUbFormershape, input1UbTailshape, false, input1TailBroadcastTiling);
        GetBroadcastTilingInfo<T>(
            tilingDataPtr->runningRank, dstUbFormershape, input2UbTailshape, false, input2TailBroadcastTiling);
    } else {
        GetBroadcastTilingInfo<T, R>(R, dstUbFormershape, input0UbFormershape, false, input0FormerBroadcastTiling);
        GetBroadcastTilingInfo<T, R>(R, dstUbFormershape, input1UbFormershape, false, input1FormerBroadcastTiling);
        GetBroadcastTilingInfo<T, R>(R, dstUbFormershape, input2UbFormershape, false, input2FormerBroadcastTiling);
        GetBroadcastTilingInfo<T, R>(R, dstUbTailshape, input0UbTailshape, false, input0TailBroadcastTiling);
        GetBroadcastTilingInfo<T, R>(R, dstUbTailshape, input1UbTailshape, false, input1TailBroadcastTiling);
        GetBroadcastTilingInfo<T, R>(R, dstUbTailshape, input2UbTailshape, false, input2TailBroadcastTiling);
    }
}

template <typename T, int64_t R>
__aicore__ inline void ClipByValueUbBroadcast<T, R>::Process()
{
    int64_t ubLoopNum =
        AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->blockTail : tilingDataPtr_->blockFormer;
    int64_t axesIndices[8] = {0};
    Ops::Base::BroadcastGetAxesIndices(
        axesIndices, tilingDataPtr_->blockFormer * AscendC::GetBlockIdx(), tilingDataPtr_->outputDims,
        tilingDataPtr_->ubSplitAxis, tilingDataPtr_->dimProductBeforeUbInner);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
        if (ubLoopIdx != 0) {
            Ops::Base::BroadcastUpdateAxesIndices(
                axesIndices, tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubOuter);
        }

        bool isUbTail = axesIndices[tilingDataPtr_->ubSplitAxis] == tilingDataPtr_->ubOuter - 1 ? true : false;

        int64_t ubSplitSize;

        int64_t input0Length;
        int64_t input1Length;
        int64_t input2Length;

        BroadcastTiling input0BroadcastTiling;
        BroadcastTiling input1BroadcastTiling;
        BroadcastTiling input2BroadcastTiling;

        uint32_t input0UbShape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t input1UbShape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t input2UbShape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t dstUbShape[8] = {1, 1, 1, 1, 1, 1, 1, 1};

        if (!isUbTail) {
            ubSplitSize = tilingDataPtr_->ubFormer;

            input0Length = input0Ublength[0];
            input1Length = input1Ublength[0];
            input2Length = input2Ublength[0];

            input0BroadcastTiling = input0FormerBroadcastTiling;
            input1BroadcastTiling = input1FormerBroadcastTiling;
            input2BroadcastTiling = input2FormerBroadcastTiling;

            for (uint32_t i = 0; i < 8; i++) {
                input0UbShape[i] = input0UbFormershape[i];
                input1UbShape[i] = input1UbFormershape[i];
                input2UbShape[i] = input2UbFormershape[i];
                dstUbShape[i] = dstUbFormershape[i];
            }
        } else {
            ubSplitSize = tilingDataPtr_->ubTail;

            input0Length = input0Ublength[1];
            input1Length = input1Ublength[1];
            input2Length = input2Ublength[1];

            input0BroadcastTiling = input0TailBroadcastTiling;
            input1BroadcastTiling = input1TailBroadcastTiling;
            input2BroadcastTiling = input2TailBroadcastTiling;

            for (uint32_t i = 0; i < 8; i++) {
                input0UbShape[i] = input0UbTailshape[i];
                input1UbShape[i] = input1UbTailshape[i];
                input2UbShape[i] = input2UbTailshape[i];
                dstUbShape[i] = dstUbTailshape[i];
            }
        }

        CopyIn0(ubSplitSize, axesIndices, ubLoopIdx, input0UbShape, dstUbShape, input0Length, input0BroadcastTiling);
        CopyIn1(ubSplitSize, axesIndices, ubLoopIdx, input1UbShape, dstUbShape, input1Length, input1BroadcastTiling);
        CopyIn2(ubSplitSize, axesIndices, ubLoopIdx, input2UbShape, dstUbShape, input2Length, input2BroadcastTiling);
        Compute3(ubSplitSize, axesIndices);
        CopyOut4(ubSplitSize, axesIndices, ubLoopIdx);
    }
}

template <typename T, int64_t R>
__aicore__ inline void ClipByValueUbBroadcast<T, R>::CopyIn0(
    int64_t ubSplitSize, const int64_t (&axesIndices)[8], int64_t ubLoopIdx, uint32_t (&srcShape)[8],
    uint32_t (&dstShape)[8], int64_t input_ub_length, BroadcastTiling& broadcastTiling)
{
    LocalTensor<T> broadcastTensor0 = broadcastTempBuf0.Get<T>();
    // 如果切分在B轴，需要做缓存处理，缓存broadcast后的数据
    if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
        (ubLoopIdx <= 1 ||
         (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
        CopyInAndBroadcast<T, R>(
            inputGmX_, broadcastTensor0, queIn0_, tilingDataPtr_->input0Strides, tilingDataPtr_->outputStrides,
            dstShape, srcShape, axesIndices, tilingDataPtr_->ubSplitAxis, input_ub_length, ubSplitSize,
            tilingDataPtr_->ubFormer, broadcastTiling);
    }
}

template <typename T, int64_t R>
__aicore__ inline void ClipByValueUbBroadcast<T, R>::CopyIn1(
    int64_t ubSplitSize, const int64_t (&axesIndices)[8], int64_t ubLoopIdx, uint32_t (&srcShape)[8],
    uint32_t (&dstShape)[8], int64_t input_ub_length, BroadcastTiling& broadcastTiling)
{
    LocalTensor<T> broadcastTensor1 = broadcastTempBuf1.Get<T>();
    // 如果切分在B轴，需要做缓存处理，缓存broadcast后的数据
    if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
        (ubLoopIdx <= 1 ||
         (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
        CopyInAndBroadcast<T, R>(
            inputGmClipValueMin_, broadcastTensor1, queIn1_, tilingDataPtr_->input1Strides,
            tilingDataPtr_->outputStrides, dstShape, srcShape, axesIndices, tilingDataPtr_->ubSplitAxis,
            input_ub_length, ubSplitSize, tilingDataPtr_->ubFormer, broadcastTiling);
    }
}

template <typename T, int64_t R>
__aicore__ inline void ClipByValueUbBroadcast<T, R>::CopyIn2(
    int64_t ubSplitSize, const int64_t (&axesIndices)[8], int64_t ubLoopIdx, uint32_t (&srcShape)[8],
    uint32_t (&dstShape)[8], int64_t input_ub_length, BroadcastTiling& broadcastTiling)
{
    LocalTensor<T> broadcastTensor2 = broadcastTempBuf2.Get<T>();
    // 如果切分在UB轴，需要做缓存处理，缓存broadcast后的数据
    if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
        (ubLoopIdx <= 1 ||
         (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
        CopyInAndBroadcast<T, R>(
            inputGmClipValueMax_, broadcastTensor2, queIn2_, tilingDataPtr_->input2Strides,
            tilingDataPtr_->outputStrides, dstShape, srcShape, axesIndices, tilingDataPtr_->ubSplitAxis,
            input_ub_length, ubSplitSize, tilingDataPtr_->ubFormer, broadcastTiling);
    }
}

template <typename T, int64_t R>
__aicore__ inline void ClipByValueUbBroadcast<T, R>::Compute3(int64_t ubSplitSize, const int64_t (&axesIndices)[8])
{
    LocalTensor<T> bufferOut0_ = queOut0_.AllocTensor<T>();
    Min(bufferOut0_, broadcastTempBuf0.Get<T>(), broadcastTempBuf2.Get<T>(),
        ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis]);
    Max(bufferOut0_, bufferOut0_, broadcastTempBuf1.Get<T>(),
        ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis]);
    queOut0_.EnQue(bufferOut0_);
}

template <typename T, int64_t R>
__aicore__ inline void ClipByValueUbBroadcast<T, R>::CopyOut4(
    int64_t ubSplitSize, const int64_t (&axesIndices)[8], int64_t ubLoopIdx)
{
    LocalTensor<T> bufferOut0_ = queOut0_.DeQue<T>();
    AscendC::DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = 1;
    dataCopyExtParams.blockLen = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] * sizeof(T);
    int64_t gmOffset = Ops::Base::BroadcastGetGmOffset(
        axesIndices, tilingDataPtr_->outputStrides, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
    AscendC::DataCopyPad(outputGmY_[gmOffset], bufferOut0_, dataCopyExtParams);
    queOut0_.FreeTensor(bufferOut0_);
}

} // namespace ClipByValue
#endif // ASCENDC_CLIP_BY_VALUEUB_BROADCAST_H_