/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cast_impl.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_CAST_IMPL_H
#define CANN_CUSTOM_OPS_CAST_IMPL_H

#include "kernel_operator.h"
namespace AscendcCast {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::RegLayout;
using AscendC::MicroAPI::SatMode;
using AscendC::MicroAPI::MaskMergeMode;

template<int N>
struct TypeGetTool;
template<>
struct TypeGetTool<CAST_TPL_BOOL> {
    using type = bool;
};
template<>
struct TypeGetTool<CAST_TPL_INT8> {
    using type = int8_t;
};
template<>
struct TypeGetTool<CAST_TPL_UINT8> {
    using type = uint8_t;
};
template<>
struct TypeGetTool<CAST_TPL_FLOAT8_E4M3FN> {
    using type = fp8_e4m3fn_t;
};
template<>
struct TypeGetTool<CAST_TPL_FLOAT8_E5M2> {
    using type = fp8_e5m2_t;
};
template<>
struct TypeGetTool<CAST_TPL_HIFLOAT8> {
    using type = hifloat8_t;
};
template<>
struct TypeGetTool<CAST_TPL_UINT16> {
    using type = uint16_t;
};
template<>
struct TypeGetTool<CAST_TPL_INT16> {
    using type = int16_t;
};
template<>
struct TypeGetTool<CAST_TPL_FLOAT16> {
    using type = half;
};
template<>
struct TypeGetTool<CAST_TPL_BF16> {
    using type = bfloat16_t;
};
template<>
struct TypeGetTool<CAST_TPL_FLOAT> {
    using type = float;
};
template<>
struct TypeGetTool<CAST_TPL_INT32> {
    using type = int32_t;
};
template<>
struct TypeGetTool<CAST_TPL_UINT32> {
    using type = uint32_t;
};
template<>
struct TypeGetTool<CAST_TPL_INT64> {
    using type = int64_t;
};
template<>
struct TypeGetTool<CAST_TPL_MAX> {
    using type = int32_t;
};

template<>
struct TypeGetTool<CAST_TPL_COMPLEX32> {
    using type = complex32;
};
template<>
struct TypeGetTool<CAST_TPL_COMPLEX64> {
    using type = complex64;
};
template<>
struct TypeGetTool<CAST_TPL_FLOAT4_E2M1> {
    using type = fp4x2_e2m1_t;
};
template<>
struct TypeGetTool<CAST_TPL_FLOAT4_E1M2> {
    using type = fp4x2_e1m2_t;
};
template<>
struct TypeGetTool<CAST_TPL_DOUBLE> {
    using type = double;
};
template<>
struct TypeGetTool<CAST_TPL_INT4> {
    using type = int4x2_t;
};

constexpr int64_t B2_BITS = 2;
constexpr int64_t B7_BITS = 7;
constexpr int64_t B8_BITS = 8;
constexpr int64_t UB_BLOCK_ALIGN = 32;
constexpr int64_t UB_BLOCK_ALIGN_MINUS_ONE = 31;
constexpr int16_t B4_MASK = 0x000F;
constexpr int16_t SHIFT_FOUR_BITS = 4;

// CAST_TEMPLATE_DIRECT_CAST
template <typename ST, typename DT>
class CastDirect {
public:
    __aicore__ inline CastDirect(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, RoundMode roundMode,
      const CastTilingData *tilingData, TPipe *pipePtr);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum_ = 2;

protected:
    __aicore__ inline void CopyIn(const int64_t &gmOffset);
    __aicore__ inline void Compute(const int64_t &len);
    __aicore__ inline void CopyOut(const int64_t &gmOffset);

protected:
    const CastTilingData *tilingData_{nullptr};
    RoundMode rMode_{RoundMode::CAST_NONE};
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<ST> xGm_;
    GlobalTensor<DT> yGm_;
    DataCopyExtParams dataCopyInParams_;
    DataCopyPadExtParams<ST> padParams_;
    DataCopyExtParams dataCopyOutParams_;
    int64_t blockIdx_{0};
};

template <typename ST, typename DT>
__aicore__ inline void CastDirect<ST, DT>::Init(GM_ADDR x, GM_ADDR y, RoundMode roundMode,
      const CastTilingData *tilingData, TPipe *pipePtr)
{
    tilingData_ = tilingData;
    pipe_ = pipePtr;
    rMode_ = roundMode;
    blockIdx_ = GetBlockIdx();

    int64_t gmBlockOffset = blockIdx_ * tilingData_->blockFormer;
    xGm_.SetGlobalBuffer((__gm__ ST *)x + gmBlockOffset);
    yGm_.SetGlobalBuffer((__gm__ DT *)y + gmBlockOffset);

    pipe_->InitBuffer(inQueueX_, bufferNum_, tilingData_->ubFormer * sizeof(ST));
    pipe_->InitBuffer(outQueue_, bufferNum_, tilingData_->ubFormer * sizeof(DT));

    dataCopyInParams_.blockCount = 1;
    dataCopyInParams_.blockLen = 0;
    dataCopyInParams_.srcStride = 0;
    dataCopyInParams_.dstStride = 0;

    dataCopyOutParams_.blockCount = 1;
    dataCopyOutParams_.blockLen = 0;
    dataCopyOutParams_.srcStride = 0;
    dataCopyOutParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;
}

template <typename ST, typename DT>
__aicore__ inline void CastDirect<ST, DT>::CopyIn(const int64_t &gmOffset)
{
    auto xLocalIn = inQueueX_.template AllocTensor<ST>();
    DataCopyPad(xLocalIn, xGm_[gmOffset], dataCopyInParams_, padParams_);
    inQueueX_.EnQue(xLocalIn);
}

template <typename ST, typename DT>
__aicore__ inline void CastDirect<ST, DT>::Compute(const int64_t &len)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    Cast<DT, ST>(yLocal, xLocal, rMode_, len);
    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <typename ST, typename DT>
__aicore__ inline void CastDirect<ST, DT>::CopyOut(const int64_t &gmOffset)
{
    auto yLocalOut = outQueue_.template DeQue<DT>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyOutParams_);
    outQueue_.FreeTensor(yLocalOut);
}

template <typename ST, typename DT>
__aicore__ inline void CastDirect<ST, DT>::Process()
{
    bool isLastBlockFlag = (blockIdx_ == tilingData_->blockNum - 1);
    int64_t loopNum = isLastBlockFlag ?
        tilingData_->ubLoopOfTailBlock : tilingData_->ubLoopOfFormerBlock;
    int64_t tailNum = isLastBlockFlag ?
        tilingData_->ubTailOfTailBlock : tilingData_->ubTailOfFormerBlock;

    int64_t gmOffset = 0;
    dataCopyInParams_.blockLen = tilingData_->ubFormer * sizeof(ST);
    dataCopyOutParams_.blockLen = tilingData_->ubFormer * sizeof(DT);
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        CopyIn(gmOffset);
        Compute(tilingData_->ubFormer);
        CopyOut(gmOffset);
        gmOffset += tilingData_->ubFormer;
    }

    dataCopyInParams_.blockLen = tailNum * sizeof(ST);
    dataCopyOutParams_.blockLen = tailNum * sizeof(DT);
    CopyIn(gmOffset);
    Compute(tailNum);
    CopyOut(gmOffset);
}

// CAST_TEMPLATE_DST_BOOL
template <typename ST>
class CastDstBool {
public:
    __aicore__ inline CastDstBool(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
      const CastTilingData *tilingData, TPipe *pipePtr);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum_ = 2;
    constexpr static ST inZero_ = 0;
protected:
    __aicore__ inline void CopyIn(const int64_t &gmOffset);
    __aicore__ inline void Compute(const int64_t &len);
    __aicore__ inline void CopyOut(const int64_t &gmOffset);

protected:
    const CastTilingData *tilingData_{nullptr};
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<> maskBuf_;
    TBuf<> boolZeroBuf_;
    GlobalTensor<ST> xGm_;
    GlobalTensor<int8_t> yGm_;
    DataCopyExtParams dataCopyInParams_;
    DataCopyPadExtParams<ST> padParams_;
    DataCopyExtParams dataCopyOutParams_;
    int64_t blockIdx_{0};
    LocalTensor<int8_t> boolZeroTensor_;
};

template <typename ST>
__aicore__ inline void CastDstBool<ST>::Init(GM_ADDR x, GM_ADDR y,
      const CastTilingData *tilingData, TPipe *pipePtr)
{
    tilingData_ = tilingData;
    pipe_ = pipePtr;
    blockIdx_ = GetBlockIdx();

    int64_t gmBlockOffset = blockIdx_ * tilingData_->blockFormer;
    xGm_.SetGlobalBuffer((__gm__ ST *)x + gmBlockOffset);
    yGm_.SetGlobalBuffer((__gm__ int8_t *)y + gmBlockOffset);

    pipe_->InitBuffer(inQueueX_, bufferNum_, tilingData_->ubFormer * sizeof(ST));
    pipe_->InitBuffer(outQueue_, bufferNum_, tilingData_->ubFormer * sizeof(int8_t));


    pipe_->InitBuffer(maskBuf_, (((tilingData_->ubFormer + B7_BITS) / B8_BITS) + UB_BLOCK_ALIGN_MINUS_ONE) /
        UB_BLOCK_ALIGN * UB_BLOCK_ALIGN);
    pipe_->InitBuffer(boolZeroBuf_, tilingData_->ubFormer * sizeof(int8_t));
    boolZeroTensor_ = boolZeroBuf_.Get<int8_t>();
    int8_t boolZero = 0;
    Duplicate(boolZeroTensor_, boolZero, tilingData_->ubFormer);

    dataCopyInParams_.blockCount = 1;
    dataCopyInParams_.blockLen = 0;
    dataCopyInParams_.srcStride = 0;
    dataCopyInParams_.dstStride = 0;

    dataCopyOutParams_.blockCount = 1;
    dataCopyOutParams_.blockLen = 0;
    dataCopyOutParams_.srcStride = 0;
    dataCopyOutParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;
}

template <typename ST>
__aicore__ inline void CastDstBool<ST>::CopyIn(const int64_t &gmOffset)
{
    auto xLocalIn = inQueueX_.template AllocTensor<ST>();
    DataCopyPad(xLocalIn, xGm_[gmOffset], dataCopyInParams_, padParams_);
    inQueueX_.EnQue(xLocalIn);
}

template <typename ST>
__aicore__ inline void CastDstBool<ST>::Compute(const int64_t &len)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto maskTensor = maskBuf_.Get<uint8_t>();
    CompareScalar(maskTensor, xLocal, inZero_, CMPMODE::EQ, len);
    inQueueX_.FreeTensor(xLocal);
    int8_t trueValue = 1;
    auto yLocal = outQueue_.template AllocTensor<int8_t>();
    Select(yLocal, maskTensor, boolZeroTensor_, trueValue, SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
    outQueue_.EnQue(yLocal);
}

template <typename ST>
__aicore__ inline void CastDstBool<ST>::CopyOut(const int64_t &gmOffset)
{
    auto yLocalOut = outQueue_.template DeQue<int8_t>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyOutParams_);
    outQueue_.FreeTensor(yLocalOut);
}

template <typename ST>
__aicore__ inline void CastDstBool<ST>::Process()
{
    bool isLastBlockFlag = (blockIdx_ == tilingData_->blockNum - 1);
    int64_t loopNum = isLastBlockFlag ?
        tilingData_->ubLoopOfTailBlock : tilingData_->ubLoopOfFormerBlock;
    int64_t tailNum = isLastBlockFlag ?
        tilingData_->ubTailOfTailBlock : tilingData_->ubTailOfFormerBlock;

    int64_t gmOffset = 0;
    dataCopyInParams_.blockLen = tilingData_->ubFormer * sizeof(ST);
    dataCopyOutParams_.blockLen = tilingData_->ubFormer * sizeof(int8_t);
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        CopyIn(gmOffset);
        Compute(tilingData_->ubFormer);
        CopyOut(gmOffset);
        gmOffset += tilingData_->ubFormer;
    }

    dataCopyInParams_.blockLen = tailNum * sizeof(ST);
    dataCopyOutParams_.blockLen = tailNum * sizeof(int8_t);
    CopyIn(gmOffset);
    Compute(tailNum);
    CopyOut(gmOffset);
}

// CAST_TEMPLATE_THROUGH
template <typename DT>
class CastThrough {
public:
    __aicore__ inline CastThrough(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
      const CastTilingData *tilingData, TPipe *pipePtr);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum_ = 2;

protected:
    __aicore__ inline void CopyIn(const int64_t &gmOffset);
    __aicore__ inline void Compute(const int64_t &len);
    __aicore__ inline void CopyOut(const int64_t &gmOffset);

protected:
    const CastTilingData *tilingData_{nullptr};
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<DT> xGm_;
    GlobalTensor<DT> yGm_;
    DataCopyExtParams dataCopyParams_;
    DataCopyPadExtParams<DT> padParams_;
    int64_t blockIdx_{0};
};

template <typename DT>
__aicore__ inline void CastThrough<DT>::Init(GM_ADDR x, GM_ADDR y,
      const CastTilingData *tilingData, TPipe *pipePtr)
{
    tilingData_ = tilingData;
    pipe_ = pipePtr;
    blockIdx_ = GetBlockIdx();

    int64_t gmBlockOffset = blockIdx_ * tilingData_->blockFormer;
    xGm_.SetGlobalBuffer((__gm__ DT *)x + gmBlockOffset);
    yGm_.SetGlobalBuffer((__gm__ DT *)y + gmBlockOffset);

    pipe_->InitBuffer(inQueueX_, bufferNum_, tilingData_->ubFormer * sizeof(DT));
    pipe_->InitBuffer(outQueue_, bufferNum_, tilingData_->ubFormer * sizeof(DT));

    dataCopyParams_.blockCount = 1;
    dataCopyParams_.blockLen = 0;
    dataCopyParams_.srcStride = 0;
    dataCopyParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;
}

template <typename DT>
__aicore__ inline void CastThrough<DT>::CopyIn(const int64_t &gmOffset)
{
    auto xLocalIn = inQueueX_.template AllocTensor<DT>();
    DataCopyPad(xLocalIn, xGm_[gmOffset], dataCopyParams_, padParams_);
    inQueueX_.EnQue(xLocalIn);
}

template <typename DT>
__aicore__ inline void CastThrough<DT>::Compute(const int64_t &len)
{
    auto xLocal = inQueueX_.template DeQue<DT>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    Copy<DT>(yLocal, xLocal, len);
    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <typename DT>
__aicore__ inline void CastThrough<DT>::CopyOut(const int64_t &gmOffset)
{
    auto yLocalOut = outQueue_.template DeQue<DT>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyParams_);
    outQueue_.FreeTensor(yLocalOut);
}

template <typename DT>
__aicore__ inline void CastThrough<DT>::Process()
{
    bool isLastBlockFlag = (blockIdx_ == tilingData_->blockNum - 1);
    int64_t loopNum = isLastBlockFlag ?
        tilingData_->ubLoopOfTailBlock : tilingData_->ubLoopOfFormerBlock;
    int64_t tailNum = isLastBlockFlag ?
        tilingData_->ubTailOfTailBlock : tilingData_->ubTailOfFormerBlock;

    int64_t gmOffset = 0;
    dataCopyParams_.blockLen = tilingData_->ubFormer * sizeof(DT);
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        CopyIn(gmOffset);
        Compute(tilingData_->ubFormer);
        CopyOut(gmOffset);
        gmOffset += tilingData_->ubFormer;
    }

    dataCopyParams_.blockLen = tailNum * sizeof(DT);
    CopyIn(gmOffset);
    Compute(tailNum);
    CopyOut(gmOffset);
}

// CAST_TEMPLATE_SRC_UINT1
template <typename DT>
class CastUint1 {
public:
    __aicore__ inline CastUint1(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
      const CastTilingData *tilingData, TPipe *pipePtr);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum_ = 2;

protected:
    __aicore__ inline void CopyIn(const int64_t &gmOffset);
    __aicore__ inline void Compute(const int64_t &len);
    __aicore__ inline void CopyOut(const int64_t &gmOffset);

protected:
    const CastTilingData *tilingData_{nullptr};
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<> oneBuf_;
    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<DT> yGm_;
    DataCopyExtParams dataCopyInParams_;
    DataCopyPadExtParams<uint8_t> padParams_;
    DataCopyExtParams dataCopyOutParams_;
    int64_t blockIdx_{0};
    LocalTensor<DT> oneTensor_;
};

template <typename DT>
__aicore__ inline void CastUint1<DT>::Init(GM_ADDR x, GM_ADDR y,
      const CastTilingData *tilingData, TPipe *pipePtr)
{
    tilingData_ = tilingData;
    pipe_ = pipePtr;
    blockIdx_ = GetBlockIdx();

    int64_t blockFormerByte = tilingData_->blockFormer / B8_BITS;
    int64_t gmInBlockOffset = blockIdx_ * blockFormerByte;
    int64_t gmOutBlockOffset = blockIdx_ * tilingData_->blockFormer;
    xGm_.SetGlobalBuffer((__gm__ uint8_t *)x + gmInBlockOffset);
    yGm_.SetGlobalBuffer((__gm__ DT *)y + gmOutBlockOffset);

    int64_t inFormerByte = tilingData_->ubFormer / B8_BITS;
    pipe_->InitBuffer(inQueueX_, bufferNum_, inFormerByte);
    pipe_->InitBuffer(outQueue_, bufferNum_, tilingData_->ubFormer * sizeof(DT));
    pipe_->InitBuffer(oneBuf_, tilingData_->ubFormer * sizeof(DT));
    oneTensor_ = oneBuf_.Get<DT>();
    DT oneValue = 1;
    Duplicate(oneTensor_, oneValue, tilingData_->ubFormer);

    dataCopyInParams_.blockCount = 1;
    dataCopyInParams_.blockLen = 0;
    dataCopyInParams_.srcStride = 0;
    dataCopyInParams_.dstStride = 0;

    dataCopyOutParams_.blockCount = 1;
    dataCopyOutParams_.blockLen = 0;
    dataCopyOutParams_.srcStride = 0;
    dataCopyOutParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;
}

template <typename DT>
__aicore__ inline void CastUint1<DT>::CopyIn(const int64_t &gmOffset)
{
    auto xLocalIn = inQueueX_.template AllocTensor<uint8_t>();
    DataCopyPad(xLocalIn, xGm_[gmOffset], dataCopyInParams_, padParams_);
    inQueueX_.EnQue(xLocalIn);
}

template <typename DT>
__aicore__ inline void CastUint1<DT>::Compute(const int64_t &len)
{
    auto xLocal = inQueueX_.template DeQue<uint8_t>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    DT zeroValue = 0;
    Select(yLocal, xLocal, oneTensor_, zeroValue, SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <typename DT>
__aicore__ inline void CastUint1<DT>::CopyOut(const int64_t &gmOffset)
{
    auto yLocalOut = outQueue_.template DeQue<DT>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyOutParams_);
    outQueue_.FreeTensor(yLocalOut);
}

template <typename DT>
__aicore__ inline void CastUint1<DT>::Process()
{
    bool isLastBlockFlag = (blockIdx_ == tilingData_->blockNum - 1);
    int64_t loopNum = isLastBlockFlag ?
        tilingData_->ubLoopOfTailBlock : tilingData_->ubLoopOfFormerBlock;
    int64_t tailNum = isLastBlockFlag ?
        tilingData_->ubTailOfTailBlock : tilingData_->ubTailOfFormerBlock;

    int64_t gmOffset = 0;
    dataCopyInParams_.blockLen = tilingData_->ubFormer / B8_BITS;
    dataCopyOutParams_.blockLen = tilingData_->ubFormer * sizeof(DT);
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        CopyIn(gmOffset);
        Compute(tilingData_->ubFormer);
        CopyOut(gmOffset);
        gmOffset += tilingData_->ubFormer;
    }

    dataCopyInParams_.blockLen = (tilingData_->ubFormer + B7_BITS) / B8_BITS;
    dataCopyOutParams_.blockLen = tailNum * sizeof(DT);
    CopyIn(gmOffset);
    Compute(tailNum);
    CopyOut(gmOffset);
}

// CAST_TEMPLATE_TWO_CAST
template <typename ST, typename MT, typename DT>
class CastTwo {
public:
    __aicore__ inline CastTwo(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, RoundMode roundMode1, RoundMode roundMode2,
      const CastTilingData *tilingData, TPipe *pipePtr);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum_ = 2;

protected:
    __aicore__ inline void CopyIn(const int64_t &gmOffset);
    __aicore__ inline void Compute(const int64_t &len);
    __aicore__ inline void CopyOut(const int64_t &gmOffset);

protected:
    const CastTilingData *tilingData_{nullptr};
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<> midTypeBuf_;
    RoundMode rMode1_{RoundMode::CAST_NONE};
    RoundMode rMode2_{RoundMode::CAST_NONE};
    GlobalTensor<ST> xGm_;
    GlobalTensor<DT> yGm_;
    DataCopyExtParams dataCopyInParams_;
    DataCopyPadExtParams<ST> padParams_;
    DataCopyExtParams dataCopyOutParams_;
    int64_t blockIdx_{0};
};

template <typename ST, typename MT, typename DT>
__aicore__ inline void CastTwo<ST, MT, DT>::Init(GM_ADDR x, GM_ADDR y, RoundMode roundMode1,
      RoundMode roundMode2, const CastTilingData *tilingData, TPipe *pipePtr)
{
    tilingData_ = tilingData;
    pipe_ = pipePtr;
    rMode1_ = roundMode1;
    rMode2_ = roundMode2;
    blockIdx_ = GetBlockIdx();

    int64_t gmBlockOffset = blockIdx_ * tilingData_->blockFormer;
    xGm_.SetGlobalBuffer((__gm__ ST *)x + gmBlockOffset);
    yGm_.SetGlobalBuffer((__gm__ DT *)y + gmBlockOffset);

    pipe_->InitBuffer(inQueueX_, bufferNum_, tilingData_->ubFormer * sizeof(ST));
    pipe_->InitBuffer(outQueue_, bufferNum_, tilingData_->ubFormer * sizeof(DT));
    pipe_->InitBuffer(midTypeBuf_, tilingData_->ubFormer * sizeof(MT));

    dataCopyInParams_.blockCount = 1;
    dataCopyInParams_.blockLen = 0;
    dataCopyInParams_.srcStride = 0;
    dataCopyInParams_.dstStride = 0;

    dataCopyOutParams_.blockCount = 1;
    dataCopyOutParams_.blockLen = 0;
    dataCopyOutParams_.srcStride = 0;
    dataCopyOutParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;
}

template <typename ST, typename MT, typename DT>
__aicore__ inline void CastTwo<ST, MT, DT>::CopyIn(const int64_t &gmOffset)
{
    auto xLocalIn = inQueueX_.template AllocTensor<ST>();
    DataCopyPad(xLocalIn, xGm_[gmOffset], dataCopyInParams_, padParams_);
    inQueueX_.EnQue(xLocalIn);
}

template <typename ST, typename MT, typename DT>
__aicore__ inline void CastTwo<ST, MT, DT>::Compute(const int64_t &len)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    LocalTensor<MT> midLocal = midTypeBuf_.Get<MT>();
    Cast<MT, ST>(midLocal, xLocal, rMode1_, len);
    inQueueX_.FreeTensor(xLocal);

    auto yLocal = outQueue_.template AllocTensor<DT>();
    if constexpr (std::is_same<MT, uint32_t>::value && std::is_same<DT, int64_t>::value) {
        LocalTensor<int32_t> tempTensor = midLocal.template ReinterpretCast<int32_t>();
        Cast<DT, int32_t>(yLocal, tempTensor, rMode1_, len);
    } else if constexpr (std::is_same<MT, uint32_t>::value && std::is_same<DT, half>::value) {
        LocalTensor<int32_t> tempTensor = midLocal.template ReinterpretCast<int32_t>();
        Cast<DT, int32_t>(yLocal, tempTensor, rMode2_, len);
    } else if constexpr (std::is_same<MT, uint32_t>::value && std::is_same<DT, float>::value) {
        LocalTensor<int32_t> tempTensor = midLocal.template ReinterpretCast<int32_t>();
        Cast<DT, int32_t>(yLocal, tempTensor, rMode2_, len);
    } else {
        Cast<DT, MT>(yLocal, midLocal, rMode2_, len);
    }
    outQueue_.EnQue(yLocal);
}

template <typename ST, typename MT, typename DT>
__aicore__ inline void CastTwo<ST, MT, DT>::CopyOut(const int64_t &gmOffset)
{
    auto yLocalOut = outQueue_.template DeQue<DT>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyOutParams_);
    outQueue_.FreeTensor(yLocalOut);
}

template <typename ST, typename MT, typename DT>
__aicore__ inline void CastTwo<ST, MT, DT>::Process()
{
    bool isLastBlockFlag = (blockIdx_ == tilingData_->blockNum - 1);
    int64_t loopNum = isLastBlockFlag ?
        tilingData_->ubLoopOfTailBlock : tilingData_->ubLoopOfFormerBlock;
    int64_t tailNum = isLastBlockFlag ?
        tilingData_->ubTailOfTailBlock : tilingData_->ubTailOfFormerBlock;

    int64_t gmOffset = 0;
    dataCopyInParams_.blockLen = tilingData_->ubFormer * sizeof(ST);
    dataCopyOutParams_.blockLen = tilingData_->ubFormer * sizeof(DT);
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        CopyIn(gmOffset);
        Compute(tilingData_->ubFormer);
        CopyOut(gmOffset);
        gmOffset += tilingData_->ubFormer;
    }

    dataCopyInParams_.blockLen = tailNum * sizeof(ST);
    dataCopyOutParams_.blockLen = tailNum * sizeof(DT);
    CopyIn(gmOffset);
    Compute(tailNum);
    CopyOut(gmOffset);
}

// CAST_TEMPLATE_MIRCRO_INOUT
// CAST_TEMPLATE_MIRCRO_CAST
// CAST_TEMPLATE_MIRCRO_CAST_INTER
// CAST_TEMPLATE_MIRCRO_CAST_DEINTER
// CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER
// CAST_TEMPLATE_MIRCRO_CAST_CAST
// CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST
// CAST_TEMPLATE_MIRCRO_CAST_DEINTER_CAST
// CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER_CAST
// CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST_CAST
// CAST_TEMPLATE_MIRCRO_DEINTER_SHIFT
template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
class CastMicro {
public:
    __aicore__ inline CastMicro(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
      const CastTilingData *tilingData, TPipe *pipePtr);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum_ = 2;

protected:
    __aicore__ inline void CopyIn(const int64_t &gmOffset);
    __aicore__ inline void Compute(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void CopyOut(const int64_t &gmOffset);

private:
    __aicore__ inline void ComputeInOut(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCast(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCastInter(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCastDeinter(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCastCastDeinter(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCastCast(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCastInterCast(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCastDeinterCast(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCastCastDeinterCast(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeCastInterCastCast(const int64_t &len, uint16_t regLoop);
    __aicore__ inline void ComputeDeinterShift(const int64_t &len, uint16_t regLoop);

protected:
    const CastTilingData *tilingData_{nullptr};
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<ST> xGm_;
    GlobalTensor<DT> yGm_;
    DataCopyExtParams dataCopyInParams_;
    DataCopyPadExtParams<ST> padParams_;
    DataCopyExtParams dataCopyOutParams_;
    int64_t blockIdx_{0};
};

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::Init(GM_ADDR x, GM_ADDR y,
    const CastTilingData *tilingData, TPipe *pipePtr)
{
    tilingData_ = tilingData;
    pipe_ = pipePtr;
    blockIdx_ = GetBlockIdx();

#if ORIG_DTYPE_X == DT_FLOAT4_E2M1 || ORIG_DTYPE_X == DT_FLOAT4_E1M2
    xGm_.SetGlobalBuffer((__gm__ ST *)x + blockIdx_ * tilingData_->blockFormer / B2_BITS);
    pipe_->InitBuffer(inQueueX_, bufferNum_, tilingData_->ubFormer * sizeof(ST) / B2_BITS);
#else
    xGm_.SetGlobalBuffer((__gm__ ST *)x + blockIdx_ * tilingData_->blockFormer);
    pipe_->InitBuffer(inQueueX_, bufferNum_, tilingData_->ubFormer * sizeof(ST));
#endif

#if ORIG_DTYPE_Y == DT_FLOAT4_E2M1 || ORIG_DTYPE_Y == DT_FLOAT4_E1M2 || ORIG_DTYPE_Y == DT_INT4
    yGm_.SetGlobalBuffer((__gm__ DT *)y + blockIdx_ * tilingData_->blockFormer / B2_BITS);
    pipe_->InitBuffer(outQueue_, bufferNum_, tilingData_->ubFormer * sizeof(DT) / B2_BITS);
#else
    yGm_.SetGlobalBuffer((__gm__ DT *)y + blockIdx_ * tilingData_->blockFormer);
    pipe_->InitBuffer(outQueue_, bufferNum_, tilingData_->ubFormer * sizeof(DT));
#endif

    dataCopyInParams_.blockCount = 1;
    dataCopyInParams_.blockLen = 0;
    dataCopyInParams_.srcStride = 0;
    dataCopyInParams_.dstStride = 0;

    dataCopyOutParams_.blockCount = 1;
    dataCopyOutParams_.blockLen = 0;
    dataCopyOutParams_.srcStride = 0;
    dataCopyOutParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::CopyIn(const int64_t &gmOffset)
{
    auto xLocalIn = inQueueX_.template AllocTensor<ST>();
    DataCopyPad(xLocalIn, xGm_[gmOffset], dataCopyInParams_, padParams_);
    inQueueX_.EnQue(xLocalIn);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::Compute(const int64_t &len, uint16_t regLoop)
{
    if constexpr (id == CAST_TEMPLATE_MIRCRO_INOUT) {
        ComputeInOut(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST) {
        ComputeCast(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST_INTER) {
        ComputeCastInter(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST_DEINTER) {
        ComputeCastDeinter(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER) {
        ComputeCastCastDeinter(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST_CAST) {
        ComputeCastCast(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST) {
        ComputeCastInterCast(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST_DEINTER_CAST) {
        ComputeCastDeinterCast(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER_CAST) {
        ComputeCastCastDeinterCast(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST_CAST) {
        ComputeCastInterCastCast(len, regLoop);
    } else if constexpr (id == CAST_TEMPLATE_MIRCRO_DEINTER_SHIFT) {
        ComputeDeinterShift(len, regLoop);
    }
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeInOut(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregTemp;
        MicroAPI::MaskReg mask;

        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            mask = MicroAPI::UpdateMask<MDT>(count);
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregTemp, srcAddr, regCopyInStep);
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                (MicroAPI::RegTensor<MDT> &)vregTemp, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCast(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);

    if constexpr (std::is_same<MST, uint32_t>::value && std::is_same<MDT, float>::value) {
        regCopyOutStep = regCopyOutStep * 2;
    }

    static constexpr MicroAPI::CastTrait trait = []() {
        if constexpr (std::is_same<MST, bfloat16_t>::value && std::is_same<MMT, int32_t>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, castMode1};
        } else if constexpr (std::is_same<MST, half>::value && std::is_same<MMT, int32_t>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
        } else if constexpr (std::is_same<MST, float>::value && std::is_same<MMT, int32_t>::value) {
            return MicroAPI::CastTrait{RegLayout::UNKNOWN, SatMode::SAT, MaskMergeMode::ZEROING, castMode1};
        } else { // 默认用fp4 转换的trait
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
        }
    }();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn;
        MicroAPI::RegTensor<MMT> vregOut;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<MST>();

        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
#if ORIG_DTYPE_X == DT_UINT32 && ORIG_DTYPE_Y == DT_FLOAT
            mask = MicroAPI::UpdateMask<int64_t>(count);
#else
            mask = MicroAPI::UpdateMask<MDT>(count);
#endif

            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn, srcAddr, regCopyInStep);
#if ORIG_DTYPE_X == DT_FLOAT4_E2M1
            MicroAPI::Cast<MMT, fp4x2_e2m1_t, trait>(vregOut, (MicroAPI::RegTensor<fp4x2_e2m1_t> &)vregIn, maskAll);
#elif ORIG_DTYPE_X == DT_FLOAT4_E1M2
            MicroAPI::Cast<MMT, fp4x2_e1m2_t, trait>(vregOut, (MicroAPI::RegTensor<fp4x2_e1m2_t> &)vregIn, maskAll);
#elif ORIG_DTYPE_X == DT_UINT32
            MicroAPI::Cast<MMT, int64_t, trait>(vregOut, (MicroAPI::RegTensor<int64_t> &)vregIn, maskAll);
#else
            MicroAPI::Cast<MMT, MST, trait>(vregOut, vregIn, maskAll);
#endif
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                (MicroAPI::RegTensor<MDT> &)vregOut, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCastInter(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    // bf16 to float
    static constexpr MicroAPI::CastTrait trait = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
    MMT zeroValue = 0;
    uint32_t count = static_cast<uint32_t>(len);

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn;
        MicroAPI::RegTensor<MMT> vregCast;
        MicroAPI::RegTensor<MMT> vregOut1;
        MicroAPI::RegTensor<MMT> vregOut2;
        MicroAPI::RegTensor<MMT> vregZero;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<MST>();
        MicroAPI::Duplicate(vregZero, zeroValue);
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn, srcAddr, regCopyInStep);
            MicroAPI::Cast<MMT, MST, trait>(vregCast, vregIn, maskAll);

            MicroAPI::Interleave<MMT>(vregOut1, vregOut2, vregCast, vregZero);

            mask = MicroAPI::UpdateMask<MDT>(count);
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                (MicroAPI::RegTensor<MDT> &)vregOut1, regCopyOutStep, mask);

            mask = MicroAPI::UpdateMask<MDT>(count);
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                (MicroAPI::RegTensor<MDT> &)vregOut2, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}


template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCastDeinter(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);

    static constexpr MicroAPI::CastTrait trait = []() {
        if constexpr (std::is_same<MST, float>::value && std::is_same<MMT, int64_t>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, castMode1};
        } else if constexpr (std::is_same<MST, float>::value && std::is_same<MMT, bfloat16_t>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, castMode1};
        }
    }();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn1;
        MicroAPI::RegTensor<MST> vregIn2;
        MicroAPI::RegTensor<MMT> vregCast1;
        MicroAPI::RegTensor<MMT> vregCast2;

        MicroAPI::RegTensor<int32_t> vregOut;
        MicroAPI::RegTensor<int32_t> vregOutNoUse;

        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<MST>();
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn1, srcAddr, regCopyInStep);
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn2, srcAddr, regCopyInStep);
            MicroAPI::Cast<MMT, MST, trait>(vregCast1, vregIn1, maskAll);
            MicroAPI::Cast<MMT, MST, trait>(vregCast2, vregIn2, maskAll);
            MicroAPI::DeInterleave<int32_t>(vregOut, vregOutNoUse,
                (MicroAPI::RegTensor<int32_t> &)vregCast1, (MicroAPI::RegTensor<int32_t> &)vregCast2);

            mask = MicroAPI::UpdateMask<MDT>(count);
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                (MicroAPI::RegTensor<MDT> &)vregOut, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCastCastDeinter(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);
    MMT zeroValue = 0;

    // bf16 to float
    static constexpr MicroAPI::CastTrait trait1 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
    // fp32 to int64
    static constexpr MicroAPI::CastTrait trait2 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, castMode2};

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn;
        MicroAPI::RegTensor<MMT> vregCast;

        MicroAPI::RegTensor<MMT> vregInter1;
        MicroAPI::RegTensor<MMT> vregInter2;


        MicroAPI::RegTensor<int64_t> vregCast1;
        MicroAPI::RegTensor<int64_t> vregCast2;
        MicroAPI::RegTensor<int32_t> vregOut;
        MicroAPI::RegTensor<int32_t> vregOutNoUse;

        MicroAPI::RegTensor<MMT> vregZero;
        MicroAPI::Duplicate(vregZero, zeroValue);

        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<int8_t>();
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn, srcAddr, regCopyInStep);
            MicroAPI::Cast<MMT, MST, trait1>(vregCast, vregIn, maskAll);
            MicroAPI::Interleave<MMT>(vregInter1, vregInter2, vregCast, vregZero);
            MicroAPI::Cast<int64_t, MMT, trait2>(vregCast1, vregInter1, maskAll);
            MicroAPI::Cast<int64_t, MMT, trait2>(vregCast2, vregInter2, maskAll);
            MicroAPI::DeInterleave<int32_t>(vregOut, vregOutNoUse,
                (MicroAPI::RegTensor<int32_t> &)vregCast1, (MicroAPI::RegTensor<int32_t> &)vregCast2);
            mask = MicroAPI::UpdateMask<MDT>(count);
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                (MicroAPI::RegTensor<MDT> &)vregOut, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCastCast(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);

    if constexpr (std::is_same<MST, uint16_t>::value && std::is_same<MDT, bfloat16_t>::value) {
        regCopyOutStep = regCopyOutStep * 2;
    }

    static constexpr MicroAPI::CastTrait trait1 = []() {
        if constexpr (std::is_same<MST, uint8_t>::value && std::is_same<MMT, bfloat16_t>::value) {
            // 当前只有fp4，统一按照fp4处理。后续如果uint8真的代表uint8，需要用编译宏区分
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
        } else if constexpr (std::is_same<MST, half>::value && std::is_same<MMT, bfloat16_t>::value) {
            return MicroAPI::CastTrait{RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
        } else if constexpr (std::is_same<MST, uint16_t>::value && std::is_same<MMT, float>::value) {
            return MicroAPI::CastTrait{RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
        }
    }();

    static constexpr MicroAPI::CastTrait trait2 = []() {
        if constexpr (std::is_same<MMT, bfloat16_t>::value && std::is_same<MDT, uint32_t>::value) {
            // 当前只有fp4，统一按照fp4处理。后续如果uint8真的代表uint8，需要用编译宏区分
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode2};
        } else if constexpr (std::is_same<MMT, bfloat16_t>::value && std::is_same<MDT, half>::value) {
            return MicroAPI::CastTrait{RegLayout::UNKNOWN, SatMode::NO_SAT, MaskMergeMode::ZEROING, castMode2};
        } else if constexpr (std::is_same<MMT, float>::value && std::is_same<MDT, bfloat16_t>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, castMode2};
        }
    }();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn;
        MicroAPI::RegTensor<MMT> vregMid;
        MicroAPI::RegTensor<MDT> vregOut;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();

        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
#if ORIG_DTYPE_X == DT_UINT16 && ORIG_DTYPE_Y == DT_BF16
            mask = MicroAPI::UpdateMask<MMT>(count);
#else
            mask = MicroAPI::UpdateMask<MDT>(count);
#endif

            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn, srcAddr, regCopyInStep);
#if ORIG_DTYPE_X == DT_FLOAT4_E2M1
            MicroAPI::Cast<MMT, fp4x2_e2m1_t, trait1>(vregMid, (MicroAPI::RegTensor<fp4x2_e2m1_t> &)vregIn, maskAll);
#elif ORIG_DTYPE_X == DT_FLOAT4_E1M2
            MicroAPI::Cast<MMT, fp4x2_e1m2_t, trait1>(vregMid, (MicroAPI::RegTensor<fp4x2_e1m2_t> &)vregIn, maskAll);
#elif ORIG_DTYPE_X == DT_UINT16
            MicroAPI::Cast<MMT, int32_t, trait1>(vregMid, (MicroAPI::RegTensor<int32_t> &)vregIn, maskAll);
#else
            MicroAPI::Cast<MMT, MST, trait1>(vregMid, vregIn, maskAll);
#endif

#if ORIG_DTYPE_Y == DT_FLOAT4_E2M1
            MicroAPI::Cast<fp4x2_e2m1_t, MMT, trait2>((MicroAPI::RegTensor<fp4x2_e2m1_t> &)vregOut, vregMid, maskAll);
#elif ORIG_DTYPE_Y == DT_FLOAT4_E1M2
            MicroAPI::Cast<fp4x2_e1m2_t, MMT, trait2>((MicroAPI::RegTensor<fp4x2_e1m2_t> &)vregOut, vregMid, maskAll);
#else
            MicroAPI::Cast<MDT, MMT, trait2>(vregOut, vregMid, maskAll);
#endif

            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                vregOut, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCastInterCast(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);
    MMT zeroValue = 0;

    static constexpr MicroAPI::CastTrait trait1 = []() {
        if constexpr (std::is_same<MST, uint8_t>::value && std::is_same<MMT, bfloat16_t>::value) {
            // 当前只有fp4，统一按照fp4处理。后续如果uint8真的代表uint8，需要用编译宏区分
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
        }
    }();

    static constexpr MicroAPI::CastTrait trait2 = []() {
        if constexpr (std::is_same<MMT, bfloat16_t>::value && std::is_same<MDT, float>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode2};
        }
    }();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn;
        MicroAPI::RegTensor<MMT> vregMid;
        MicroAPI::RegTensor<MDT> vregOut1;
        MicroAPI::RegTensor<MDT> vregOut2;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::RegTensor<MMT> vregZero;
        MicroAPI::Duplicate(vregZero, zeroValue);
        MicroAPI::RegTensor<MMT> vregInter1;
        MicroAPI::RegTensor<MMT> vregInter2;

        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn, srcAddr, regCopyInStep);
#if ORIG_DTYPE_X == DT_FLOAT4_E2M1
            MicroAPI::Cast<MMT, fp4x2_e2m1_t, trait1>(vregMid, (MicroAPI::RegTensor<fp4x2_e2m1_t> &)vregIn, maskAll);
#elif ORIG_DTYPE_X == DT_FLOAT4_E1M2
            MicroAPI::Cast<MMT, fp4x2_e1m2_t, trait1>(vregMid, (MicroAPI::RegTensor<fp4x2_e1m2_t> &)vregIn, maskAll);
#else
            MicroAPI::Cast<MMT, MST, trait1>(vregMid, vregIn, maskAll);
#endif
            MicroAPI::Interleave<MMT>(vregInter1, vregInter2, vregMid, vregZero);
            MicroAPI::Cast<MDT, MMT, trait2>(vregOut1, vregInter1, maskAll);
            MicroAPI::Cast<MDT, MMT, trait2>(vregOut2, vregInter2, maskAll);

            mask = MicroAPI::UpdateMask<MDT>(count);
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                vregOut1, regCopyOutStep, mask);
            mask = MicroAPI::UpdateMask<MDT>(count);
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                vregOut2, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCastDeinterCast(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);

    if constexpr ((std::is_same<MST, uint32_t>::value && std::is_same<MDT, half>::value) ||  
                  (std::is_same<MST, uint32_t>::value && std::is_same<MDT, bfloat16_t>::value)) {
        regCopyOutStep = regCopyOutStep * 2;
    }

    static constexpr MicroAPI::CastTrait trait1 = []() {
        if constexpr (std::is_same<MST, float>::value && std::is_same<MMT, bfloat16_t>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, castMode1};
        } else if constexpr (std::is_same<MST, uint32_t>::value && std::is_same<MMT, float>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode1};
        }
    }();

    static constexpr MicroAPI::CastTrait trait2 = []() {
        if constexpr (std::is_same<MMT, bfloat16_t>::value && std::is_same<MDT, uint32_t>::value) {
            // 当前只有fp4，统一按照fp4处理。后续如果uint8真的代表uint8，需要用编译宏区分
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, castMode2};
        } else if constexpr (std::is_same<MMT, float>::value && std::is_same<MDT, half>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, castMode2};
        } else if constexpr (std::is_same<MMT, float>::value && std::is_same<MDT, bfloat16_t>::value) {
            return MicroAPI::CastTrait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, castMode2};
        }
    }();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn1;
        MicroAPI::RegTensor<MST> vregIn2;
        MicroAPI::RegTensor<MMT> vregMid1;
        MicroAPI::RegTensor<MMT> vregMid2;
        MicroAPI::RegTensor<MMT> vregDeinter;
        MicroAPI::RegTensor<MMT> vregDeinterNoUse;
        MicroAPI::RegTensor<MDT> vregOut;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();

        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn1, srcAddr, regCopyInStep);
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn2, srcAddr, regCopyInStep);
#if ORIG_DTYPE_X == DT_UINT32
            mask = MicroAPI::UpdateMask<MMT>(count);
            MicroAPI::Cast<MMT, int64_t, trait1>(vregMid1, (MicroAPI::RegTensor<int64_t> &)vregIn1, maskAll);
            MicroAPI::Cast<MMT, int64_t, trait1>(vregMid2, (MicroAPI::RegTensor<int64_t> &)vregIn2, maskAll);
#else
            mask = MicroAPI::UpdateMask<MDT>(count);
            MicroAPI::Cast<MMT, MST, trait1>(vregMid1, vregIn1, maskAll);
            MicroAPI::Cast<MMT, MST, trait1>(vregMid2, vregIn2, maskAll);
#endif
        
            MicroAPI::DeInterleave<MMT>(vregDeinter, vregDeinterNoUse, vregMid1, vregMid2);
#if ORIG_DTYPE_Y == DT_FLOAT4_E2M1
            MicroAPI::Cast<fp4x2_e2m1_t, MMT, trait2>((MicroAPI::RegTensor<fp4x2_e2m1_t> &)vregOut, vregDeinter, maskAll);
#elif ORIG_DTYPE_Y == DT_FLOAT4_E1M2
            MicroAPI::Cast<fp4x2_e1m2_t, MMT, trait2>((MicroAPI::RegTensor<fp4x2_e1m2_t> &)vregOut, vregDeinter, maskAll);
#else
            MicroAPI::Cast<MDT, MMT, trait2>(vregOut, vregDeinter, maskAll);
#endif
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                vregOut, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCastCastDeinterCast(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);

    // fp8 to fp32
    static constexpr MicroAPI::CastTrait trait1 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    // fp32 to bf16
    static constexpr MicroAPI::CastTrait trait2 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    // bf16 to fp4
    static constexpr MicroAPI::CastTrait trait3 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn1;
        MicroAPI::RegTensor<MST> vregIn2;
        MicroAPI::RegTensor<MMT> vregMid1;
        MicroAPI::RegTensor<MMT> vregMid2;
        MicroAPI::RegTensor<bfloat16_t> vregMidBf1;
        MicroAPI::RegTensor<bfloat16_t> vregMidBf2;
        MicroAPI::RegTensor<bfloat16_t> vregDeinter;
        MicroAPI::RegTensor<bfloat16_t> vregDeinterNoUse;
        MicroAPI::RegTensor<MDT> vregOut;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();

        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn1, srcAddr, regCopyInStep);
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn2, srcAddr, regCopyInStep);
            MicroAPI::Cast<MMT, MST, trait1>(vregMid1, vregIn1, maskAll);
            MicroAPI::Cast<MMT, MST, trait1>(vregMid2, vregIn2, maskAll);
            MicroAPI::Cast<bfloat16_t, MMT, trait2>(vregMidBf1, vregMid1, maskAll);
            MicroAPI::Cast<bfloat16_t, MMT, trait2>(vregMidBf2, vregMid2, maskAll);
            MicroAPI::DeInterleave<bfloat16_t>(vregDeinter, vregDeinterNoUse, vregMidBf1, vregMidBf2);
#if ORIG_DTYPE_Y == DT_FLOAT4_E2M1
            MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, trait3>((MicroAPI::RegTensor<fp4x2_e2m1_t> &)vregOut, vregDeinter, maskAll);
#elif ORIG_DTYPE_Y == DT_FLOAT4_E1M2
            MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, trait3>((MicroAPI::RegTensor<fp4x2_e1m2_t> &)vregOut, vregDeinter, maskAll);
#else
            MicroAPI::Cast<MDT, bfloat16_t, trait3>(vregOut, vregDeinter, maskAll);
#endif
            // 输入fp8，ub former按照VL/sizeof(input)算的对齐，输出的ub buffer按照former分配内存，足够大，
            // 而且拷出使用的是4字节转1字节转出，实际每次VF循环数据量是VL/4 *2，所以拷出ub可以直接用maskAll
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                vregOut, regCopyOutStep, maskAll);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeCastInterCastCast(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);
    bfloat16_t zeroValue = 0;

    // fp4 to bf16
    static constexpr MicroAPI::CastTrait trait1 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    // bf16 to fp32
    static constexpr MicroAPI::CastTrait trait2 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    // fp32 to fp8
#if ORIG_DTYPE_Y == DT_HIFLOAT8
    static constexpr MicroAPI::CastTrait trait3 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
#else
    static constexpr MicroAPI::CastTrait trait3 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
#endif

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn;
        MicroAPI::RegTensor<bfloat16_t> vregMidBf;
        MicroAPI::RegTensor<MMT> vregMid1;
        MicroAPI::RegTensor<MMT> vregMid2;
        MicroAPI::RegTensor<MDT> vregOut1;
        MicroAPI::RegTensor<MDT> vregOut2;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::RegTensor<bfloat16_t> vregZero;
        MicroAPI::Duplicate(vregZero, zeroValue);
        MicroAPI::RegTensor<bfloat16_t> vregInter1;
        MicroAPI::RegTensor<bfloat16_t> vregInter2;

        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn, srcAddr, regCopyInStep);
#if ORIG_DTYPE_X == DT_FLOAT4_E2M1
            MicroAPI::Cast<bfloat16_t, fp4x2_e2m1_t, trait1>(vregMidBf, (MicroAPI::RegTensor<fp4x2_e2m1_t> &)vregIn, maskAll);
#elif ORIG_DTYPE_X == DT_FLOAT4_E1M2
            MicroAPI::Cast<bfloat16_t, fp4x2_e1m2_t, trait1>(vregMidBf, (MicroAPI::RegTensor<fp4x2_e1m2_t> &)vregIn, maskAll);
#else
            MicroAPI::Cast<bfloat16_t, MST, trait1>(vregMidBf, vregIn, maskAll);
#endif
            MicroAPI::Interleave<bfloat16_t>(vregInter1, vregInter2, vregMidBf, vregZero);
            MicroAPI::Cast<MMT, bfloat16_t, trait2>(vregMid1, vregInter1, maskAll);
            MicroAPI::Cast<MMT, bfloat16_t, trait2>(vregMid2, vregInter2, maskAll);
#if ORIG_DTYPE_Y == DT_HIFLOAT8
            MicroAPI::Cast<hifloat8_t, MMT, trait3>((MicroAPI::RegTensor<hifloat8_t> &)vregOut1, vregMid1, maskAll);
            MicroAPI::Cast<hifloat8_t, MMT, trait3>((MicroAPI::RegTensor<hifloat8_t> &)vregOut2, vregMid2, maskAll);
#elif ORIG_DTYPE_Y == DT_FLOAT8_E5M2
            MicroAPI::Cast<fp8_e5m2_t, MMT, trait3>((MicroAPI::RegTensor<fp8_e5m2_t> &)vregOut1, vregMid1, maskAll);
            MicroAPI::Cast<fp8_e5m2_t, MMT, trait3>((MicroAPI::RegTensor<fp8_e5m2_t> &)vregOut2, vregMid2, maskAll);
#elif ORIG_DTYPE_Y == DT_FLOAT8_E4M3FN
            MicroAPI::Cast<fp8_e4m3fn_t, MMT, trait3>((MicroAPI::RegTensor<fp8_e4m3fn_t> &)vregOut1, vregMid1, maskAll);
            MicroAPI::Cast<fp8_e4m3fn_t, MMT, trait3>((MicroAPI::RegTensor<fp8_e4m3fn_t> &)vregOut2, vregMid2, maskAll);
#endif
            // 输入fp4，ub former按照VL/sizeof(input)算的对齐，输出的ub buffer按照former分配内存，足够大，
            // 而且拷出使用的是4字节转1字节转出，实际每次VF循环数据量是VL/4 *2，所以拷出ub可以直接用maskAll
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                vregOut1, regCopyOutStep, maskAll);
            MicroAPI::DataCopy<MDT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                vregOut2, regCopyOutStep, maskAll);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::ComputeDeinterShift(const int64_t &len, uint16_t regLoop)
{
    auto xLocal = inQueueX_.template DeQue<ST>();
    auto yLocal = outQueue_.template AllocTensor<DT>();
    __ubuf__ MST *srcAddr = (__ubuf__ MST *)xLocal.GetPhyAddr();
    __ubuf__ MDT *dstAddr = (__ubuf__ MDT *)yLocal.GetPhyAddr();
    int32_t regCopyInStep = static_cast<int32_t>(tilingData_->regCopyInStep);
    int32_t regCopyOutStep = static_cast<int32_t>(tilingData_->regCopyOutStep);
    uint32_t count = static_cast<uint32_t>(len);

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<MST> vregIn1;
        MicroAPI::RegTensor<MST> vregIn2;
        MicroAPI::RegTensor<MMT> vregDeinter1;
        MicroAPI::RegTensor<MMT> vregDeinter2;
        MicroAPI::RegTensor<MMT> vregMid1;
        MicroAPI::RegTensor<MMT> vregMid2;
        MicroAPI::RegTensor<MMT> vreg4Bit;
        MicroAPI::RegTensor<MMT> vregOut;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::Duplicate(vreg4Bit, B4_MASK);

        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn1, srcAddr, regCopyInStep);
            MicroAPI::DataCopy<MST, MicroAPI::PostLiteral::POST_MODE_UPDATE, ldDist>(vregIn2, srcAddr, regCopyInStep);
        
            MicroAPI::DeInterleave<MMT>(vregDeinter1, vregDeinter2, vregIn1, vregIn2);
            MicroAPI::And<MMT, MicroAPI::MaskMergeMode::ZEROING>(vregMid1, vregDeinter1, vreg4Bit, maskAll);
            MicroAPI::ShiftLefts(vregMid2, vregDeinter2, SHIFT_FOUR_BITS, maskAll);
            MicroAPI::Or<MMT, MicroAPI::MaskMergeMode::ZEROING>(vregOut, vregMid1, vregMid2, maskAll);

            mask = MicroAPI::UpdateMask<MMT>(count);
            MicroAPI::DataCopy<MMT, MicroAPI::PostLiteral::POST_MODE_UPDATE, stDist>(dstAddr,
                vregOut, regCopyOutStep, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::CopyOut(const int64_t &gmOffset)
{
    auto yLocalOut = outQueue_.template DeQue<DT>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyOutParams_);
    outQueue_.FreeTensor(yLocalOut);
}

template <int id, typename ST, typename DT, typename MST, typename MMT, typename MDT,
    LoadDist ldDist, StoreDist stDist, RoundMode castMode1, RoundMode castMode2>
__aicore__ inline void CastMicro<id, ST, DT, MST, MMT, MDT,
    ldDist, stDist, castMode1, castMode2>::Process()
{
    bool isLastBlockFlag = (blockIdx_ == tilingData_->blockNum - 1);
    int64_t loopNum = isLastBlockFlag ?
        tilingData_->ubLoopOfTailBlock : tilingData_->ubLoopOfFormerBlock;
    int64_t tailNum = isLastBlockFlag ?
        tilingData_->ubTailOfTailBlock : tilingData_->ubTailOfFormerBlock;
    int64_t tailRegLoop = isLastBlockFlag ?
        tilingData_->ubTailOfTailRegLoop : tilingData_->ubTailOfFormerRegLoop;


#if ORIG_DTYPE_X == DT_FLOAT4_E2M1 || ORIG_DTYPE_X == DT_FLOAT4_E1M2
    int64_t xRealFormer = tilingData_->ubFormer / B2_BITS;
#else
    int64_t xRealFormer = tilingData_->ubFormer;
#endif

#if ORIG_DTYPE_Y == DT_FLOAT4_E2M1 || ORIG_DTYPE_Y == DT_FLOAT4_E1M2 || ORIG_DTYPE_Y == DT_INT4
    int64_t yRealFormer = tilingData_->ubFormer / B2_BITS;
#else
    int64_t yRealFormer = tilingData_->ubFormer;
#endif

    dataCopyInParams_.blockLen = xRealFormer * sizeof(ST);
    dataCopyOutParams_.blockLen = yRealFormer * sizeof(DT);
    int64_t xGmOffset = 0;
    int64_t yGmOffset = 0;
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        CopyIn(xGmOffset);
        Compute(tilingData_->ubFormer, static_cast<uint16_t>(tilingData_->ubFormerRegLoop));
        CopyOut(yGmOffset);
        xGmOffset += xRealFormer;
        yGmOffset += yRealFormer;
    }

#if ORIG_DTYPE_X == DT_FLOAT4_E2M1 || ORIG_DTYPE_X == DT_FLOAT4_E1M2
    dataCopyInParams_.blockLen = (tailNum + 1) / B2_BITS * sizeof(ST);
#else
    dataCopyInParams_.blockLen = tailNum * sizeof(ST);
#endif

#if ORIG_DTYPE_Y == DT_FLOAT4_E2M1 || ORIG_DTYPE_Y == DT_FLOAT4_E1M2 || ORIG_DTYPE_Y == DT_INT4
    dataCopyOutParams_.blockLen = (tailNum + 1) / B2_BITS * sizeof(DT);
#else
    dataCopyOutParams_.blockLen = tailNum * sizeof(DT);
#endif

    CopyIn(xGmOffset);
    Compute(tailNum, static_cast<uint16_t>(tailRegLoop));
    CopyOut(yGmOffset);
}

} // namespace AscendcCast
#endif  // CANN_CUSTOM_OPS_CAST_IMPL_H