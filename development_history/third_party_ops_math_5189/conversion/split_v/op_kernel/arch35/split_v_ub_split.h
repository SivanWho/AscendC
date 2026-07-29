/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_UB_SPLIT_H
#define SPLIT_V_UB_SPLIT_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SplitV {
using namespace AscendC;

const int32_t DATA_COPY_GATHER = 128; // DataCopyGather并行度
const int32_t SPLIT_UB_NUM = 4;       // 输入输出均开db
const int32_t INT_NUM_TWO = 2;
const int32_t MAX_COPY_BYTE = 1024;
const int32_t HUGE_NUM_SPLIT = 1024 * 4;
const int32_t MORE_M = 64;
const int32_t ONE_BLOCK_UB = Ops::Base::GetUbBlockSize(); // 一个block的ub大小

template <typename T, typename U, typename Y> // T原始数据类型  U做datacopygather的数据类型 Y是做vci的数据类型
class SplitVUbSplit {
public:
    __aicore__ inline SplitVUbSplit(TPipe &pipe) : pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR sizeSplits, const SplitVTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DataCopyGatherVf(int64_t mFactor, int64_t niSize, int64_t ubStartOffset, int64_t nUbFactor,
        LocalTensor<T> &srcUbSize, int64_t ubOffset);
    __aicore__ inline void DataCopyVf(int64_t mFactor, int64_t niSize, int64_t ubStartOffset, int64_t curNFactor,
        LocalTensor<T> &srcUbSize, int64_t ubOffset);
    __aicore__ inline __gm__ T *GetTensorAddr(int64_t index, int64_t offset);
    __aicore__ inline int64_t SplitPrefix(int64_t index);
    __aicore__ inline int64_t CurSplitSize(int64_t index);
    __aicore__ inline void ComputeEndLine(int32_t &curNFactor, int64_t nOffset, int64_t nowBlockEnd);
    __aicore__ inline void DataCopyIn(int64_t mOffset,int64_t nOffset, int32_t curNFactor,
        int32_t mUbFactorNow, LocalTensor<T> &xUb);
    __aicore__ inline void UbProcess(int32_t mTimes, int32_t mUbFactor, int64_t nBlockFactorNow,
       int64_t mBlockFactorNow, int32_t nUbFactor, int64_t nowBlockEnd);
    __aicore__ inline void MoreCopyOut(int32_t curNFactor,int32_t curNFactorBeginSplit,
        int32_t curNFactorEndSplit, int64_t mOffset, int64_t nOffset,
        int64_t perSize, int32_t mUbFactorNow, LocalTensor<T> &xUb);
    __aicore__ inline void OnceCopyOut(int32_t curNFactorBeginSplit, int64_t mOffset, int64_t nOffset,
        int64_t perSize, int32_t curNFactor, int32_t mUbFactorNow, LocalTensor<T> &xUb);
    __aicore__ inline void DataCopyOutGm(int32_t blockCount, int64_t blockLen, int64_t stride, int64_t ubOffset);
    __aicore__ inline void AllCopyOut(int64_t mOffset,int64_t nOffset, int32_t &curNFactor,
        int32_t mUbFactorNow, int64_t nowBlockEnd, LocalTensor<T> &xUb);

private:
    TPipe &pipe_;
    constexpr static int32_t bufferNum = 2;
    constexpr static int64_t BLOCK_ELENUM = Ops::Base::GetUbBlockSize() / sizeof(T);
    constexpr static int64_t MAX_COPY_NUM = MAX_COPY_BYTE / sizeof(T);
    const SplitVTilingData *tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueY_;
    TBuf<> splitBuf_;
    TBuf<> splitBufInt32_;
    LocalTensor<T> yLocal_;
    LocalTensor<int64_t> splitOffsetLocal_;
    LocalTensor<int32_t> splitOffsetLocalInt32_;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<int64_t> sizeSplitsGm_;
    GlobalTensor<int32_t> sizeSplitsGmInt32_;
    ListTensorDesc inputList_;
    int32_t blockIdx_ = 0;
    int32_t oneUbNum_ = ONE_BLOCK_UB / sizeof(T);
    int32_t ubSizeNum_ = 0;
    int64_t nBlockOffset_ = 0;
    int64_t mBlockOffset_ = 0;
    int32_t maxGatherNum_ = DATA_COPY_GATHER / sizeof(T);
    int32_t vfLen_ = Ops::Base::GetVRegSize() / sizeof(T);
    int32_t vfLenU_ = Ops::Base::GetVRegSize() / sizeof(U);
    int32_t numSplit_ = 0;
    int64_t splitPrefixNow_ = 0;
    int64_t splitPrefixNext_ = 0;
    int64_t splitEndLine_ = 0;
    int64_t curProcessSplit_ = 0;
    int64_t curNOffset_ = 0;
    int64_t curSplitOffset_ = 0;

    DataCopyPadExtParams<T> padParams = { false, 0, 0, 0 };
};

template <typename T1, typename T2>
__aicore__ inline void DataCopyGatherScope(int32_t vfLenU, int32_t ubSizeNum, int64_t mFactor, int64_t niSize,
    int64_t ubStartOffset,int64_t nUbFactor, LocalTensor<T1> &srcUbSize, LocalTensor<T1> &dstUbSize, int64_t ubOffset)
{
    uint32_t size0 = vfLenU / (niSize);
    if (size0 > mFactor) {
        size0 = mFactor;
    }
    uint32_t offset = size0 * nUbFactor;
    uint32_t num = size0*niSize;
    uint16_t times = (mFactor - size0) / size0;
    uint32_t tail = (mFactor - size0 - times*size0)*niSize;
    uint32_t mask1 = num;
    uint32_t mask2 = tail;
    __ubuf__ T1 *srcPtr = (__ubuf__ T1 *)srcUbSize.GetPhyAddr();
    __ubuf__ T1 *dstPtr = (__ubuf__ T1 *)dstUbSize.GetPhyAddr() + ubOffset;
    __ubuf__ T1 *curDstPtr = dstPtr + num;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::RegTensor<T2> indexReg;
        AscendC::MicroAPI::RegTensor<T1> tmp;
        AscendC::MicroAPI::RegTensor<T1> addReg;
        AscendC::MicroAPI::RegTensor<T1> addReg1;
        AscendC::MicroAPI::RegTensor<T1> dstReg;
        AscendC::MicroAPI::RegTensor<T1> tmp1;
        AscendC::MicroAPI::RegTensor<T1> tmp2;
        AscendC::MicroAPI::RegTensor<T1> subReg;
        AscendC::MicroAPI::RegTensor<T1> niReg;
        AscendC::MicroAPI::UnalignReg u0;
        p0 = AscendC::MicroAPI::UpdateMask<T1>(mask1);
        AscendC::MicroAPI::Duplicate(niReg, (T1)niSize, p0);
        AscendC::MicroAPI::Arange(indexReg, 0);
        AscendC::MicroAPI::Div(tmp, (AscendC::MicroAPI::RegTensor<T1> &)indexReg, niReg, p0);
        AscendC::MicroAPI::Muls(tmp1, tmp, (T1)nUbFactor, p0);
        AscendC::MicroAPI::Mul(subReg, tmp, niReg, p0);
        AscendC::MicroAPI::Sub(tmp2, (AscendC::MicroAPI::RegTensor<T1> &)indexReg, subReg, p0);
        AscendC::MicroAPI::Add(addReg, tmp1, tmp2, p0);
        AscendC::MicroAPI::Adds(addReg, addReg, (T1)ubStartOffset, p0);
        AscendC::MicroAPI::DataCopyGather(dstReg, srcPtr, addReg, p0);
        AscendC::MicroAPI::DataCopy(dstPtr, dstReg, p0);
        for (uint16_t ii = 0; ii < times; ii++) {
            AscendC::MicroAPI::Adds(addReg1, addReg, (T1)(offset*(ii + 1)), p0);
            AscendC::MicroAPI::DataCopyGather(dstReg, srcPtr, addReg1, p0);
            AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstReg, u0, num);
            AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
        }
        p1 = AscendC::MicroAPI::UpdateMask<T1>(mask2);
        AscendC::MicroAPI::Adds(addReg1, addReg, (T1)(offset*(times + 1)), p1);
        AscendC::MicroAPI::DataCopyGather(dstReg, srcPtr, addReg1, p1);
        AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstReg, u0, tail);
        AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::DataCopyGatherVf(int64_t mFactor, int64_t niSize,
    int64_t ubStartOffset, int64_t nUbFactor, LocalTensor<T> &srcUbSize, int64_t ubOffset)
{
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        LocalTensor<U> srcUbSizeInt32 = srcUbSize.template ReinterpretCast<U>();
        LocalTensor<U> dstUbSizeInt32 = yLocal_.template ReinterpretCast<U>();
        DataCopyGatherScope<U, Y>(vfLenU_, ubOffset * 2, mFactor, niSize * 2,
                                  ubStartOffset * 2, nUbFactor * 2, srcUbSizeInt32, dstUbSizeInt32, ubOffset * 2);
    } else {
        uint32_t ubSizeNum = ubOffset;
        uint32_t size0 = vfLenU_ / niSize;
        if (size0 > mFactor) {
           size0 = mFactor;
        }
        uint32_t offset = size0 * nUbFactor;
        uint32_t num = size0*niSize;
        uint16_t times = (mFactor - size0) / size0;
        uint32_t tail = (mFactor - size0 - times*size0)*niSize;
        uint32_t mask1 = num;
        uint32_t mask2 = tail;
        if constexpr (sizeof(T) == sizeof(int8_t)) {
            ubSizeNum = ubSizeNum / 2;
        }
        __ubuf__ T *srcPtr = (__ubuf__ T *)srcUbSize.GetPhyAddr();
        __ubuf__ U *dstPtr = (__ubuf__ U *)yLocal_.GetPhyAddr() + ubSizeNum;
        __ubuf__ T *curDstPtr = (__ubuf__ T *)dstPtr + num;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg p0;
            AscendC::MicroAPI::MaskReg p1;
            AscendC::MicroAPI::RegTensor<Y> indexReg;
            AscendC::MicroAPI::RegTensor<U> tmp;
            AscendC::MicroAPI::RegTensor<U> addReg;
            AscendC::MicroAPI::RegTensor<U> addReg1;
            AscendC::MicroAPI::RegTensor<U> dstReg;
            AscendC::MicroAPI::RegTensor<U> tmp1;
            AscendC::MicroAPI::RegTensor<U> tmp2;
            AscendC::MicroAPI::RegTensor<U> subReg;
            AscendC::MicroAPI::RegTensor<U> niReg;
            AscendC::MicroAPI::RegTensor<T> dstRegO;
            AscendC::MicroAPI::UnalignReg u0;
            p0 = AscendC::MicroAPI::UpdateMask<U>(mask1);
            AscendC::MicroAPI::Duplicate(niReg, (U)niSize, p0);
            AscendC::MicroAPI::Arange(indexReg, 0);
            AscendC::MicroAPI::Div(tmp, (AscendC::MicroAPI::RegTensor<U> &)indexReg, niReg, p0);
            AscendC::MicroAPI::Muls(tmp1, tmp, (U)nUbFactor, p0);
            AscendC::MicroAPI::Mul(subReg, tmp, niReg, p0);
            AscendC::MicroAPI::Sub(tmp2, (AscendC::MicroAPI::RegTensor<U> &)indexReg, subReg, p0);
            AscendC::MicroAPI::Add(addReg, tmp1, tmp2, p0);
            AscendC::MicroAPI::Adds(addReg, addReg, (U)ubStartOffset, p0);
            AscendC::MicroAPI::DataCopyGather(dstReg, srcPtr, addReg, p0);
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                    dstPtr, dstReg, p0);
            } else {
                AscendC::MicroAPI::DataCopy(dstPtr, dstReg, p0);
            }
            for (uint16_t ii = 0; ii < times; ii++) {
                AscendC::MicroAPI::Adds(addReg1, addReg, (U)(offset*(ii + 1)), p0);
                AscendC::MicroAPI::DataCopyGather(dstReg, srcPtr, addReg1, p0);
                if constexpr (sizeof(T) == sizeof(int8_t)) {
                    AscendC::MicroAPI::Pack(dstRegO, dstReg);
                    AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstRegO, u0, num);
                    AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
                } else {
                    AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstReg, u0, num);
                    AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
                }
            }
            p1 = AscendC::MicroAPI::UpdateMask<U>(mask2);
            AscendC::MicroAPI::Adds(addReg1, addReg, (U)(offset*(times + 1)), p1);
            AscendC::MicroAPI::DataCopyGather(dstReg, srcPtr, addReg1, p1);
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                AscendC::MicroAPI::Pack(dstRegO, dstReg);
                AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstRegO, u0, tail);
                AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
            } else {
                AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstReg, u0, tail);
                AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
            }
        }
    }
}

template <typename T1>
__aicore__ inline void DataCopyScope(uint16_t size0, int64_t offset, int64_t nisize,
    uint32_t vfLen1 ,__ubuf__ T1* srcPtr, __ubuf__ T1* dstPtr)
{
    uint16_t repeatTimes = nisize / vfLen1;
    uint32_t tail = nisize - repeatTimes*vfLen1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T1> vd0;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < size0; i++) {
            __ubuf__ T1* srcPtr1 = srcPtr + i * offset;
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcPtr1);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vd0, u0,
                    srcPtr1, vfLen1);
                AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstPtr, vd0,
                    u1, vfLen1);
            }
            AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vd0, u0,
                srcPtr1, vfLen1);
            AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstPtr, vd0,
                u1, tail);
            AscendC::MicroAPI::DataCopyUnAlignPost<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstPtr, u1, 0);
        }
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::DataCopyVf(int64_t mFactor, int64_t niSize, int64_t ubStartOffset,
    int64_t curNFactor, LocalTensor<T> &srcUbSize, int64_t ubOffset)
{
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        __ubuf__ U *srcPtr = (__ubuf__ U *)srcUbSize.GetPhyAddr() + ubStartOffset*2;
        __ubuf__ U *dstPtr = (__ubuf__ U *)yLocal_.GetPhyAddr() + ubOffset*2;
        int64_t offset = curNFactor*2;
        DataCopyScope<U>((uint16_t)mFactor, offset,niSize*2, vfLenU_, srcPtr, dstPtr);
    } else {
        __ubuf__ T *srcPtr = (__ubuf__ T *)srcUbSize.GetPhyAddr() + ubStartOffset;
        __ubuf__ T *dstPtr = (__ubuf__ T *)yLocal_.GetPhyAddr() + ubOffset;
        DataCopyScope<T>((uint16_t)mFactor, curNFactor, niSize, vfLen_, srcPtr, dstPtr);
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline __gm__ T *SplitVUbSplit<T, U, Y>::GetTensorAddr(int64_t index, int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, typename U, typename Y>
__aicore__ inline int64_t SplitVUbSplit<T, U, Y>::SplitPrefix(int64_t index)
{
    int64_t tensorSize = 0;
    for (int64_t jj = 0; jj < index + 1; jj++) {
        if (jj >= 0 && jj < numSplit_) {
            tensorSize = tensorSize + splitOffsetLocal_.GetValue(jj);
        }
    }
    return tensorSize;
}

template <typename T, typename U, typename Y>
__aicore__ inline int64_t SplitVUbSplit<T, U, Y>::CurSplitSize(int64_t index)
{
    int64_t tensorSize = 0;
    if (index >=0 && index < numSplit_) {
        tensorSize = splitOffsetLocal_.GetValue(index);
    }

    return tensorSize;
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::ComputeEndLine(int32_t &curNFactor, int64_t nOffset, int64_t nowBlockEnd)
{
    int64_t endLine = nOffset + nBlockOffset_ + curNFactor;
    bool isBoundary = (splitEndLine_ == curSplitOffset_ || endLine >= nowBlockEnd) ? true : false;
    if (isBoundary) {
        return;
    } else {
        int64_t rightEnd = curSplitOffset_;
        if (rightEnd > nowBlockEnd) {
            rightEnd = nowBlockEnd;
        }
        int64_t endLineSuffixR = rightEnd - endLine;
        int64_t beforeSize = 0;
        if (curProcessSplit_ - 1 >= 0) {
            beforeSize = curSplitOffset_ - CurSplitSize(curProcessSplit_);
        }
        int64_t endLinePreL = endLine - beforeSize;     // endLine左边的长度
        int64_t moveLines = oneUbNum_ - endLineSuffixR; // 满足右半部分满32b, 需要移动的长度
        int64_t shengyuL = endLinePreL - moveLines;
        if (endLineSuffixR >= oneUbNum_ && endLinePreL >= oneUbNum_) {
            // 无需调整
            return;
        } else if ((endLineSuffixR >= oneUbNum_ && endLinePreL < oneUbNum_) ||
            (endLineSuffixR < oneUbNum_ && (shengyuL <= 0 || (shengyuL > 0 && shengyuL < oneUbNum_)))) {
            // 左半部分压缩为0
            curNFactor = curNFactor - endLinePreL;
            splitEndLine_ -= endLinePreL;
            curProcessSplit_ -= 1;
            curSplitOffset_ = splitEndLine_;
            return;
        } else {
            // 移动后，左边长度大于等于32b, 不再移动
            curNFactor = curNFactor - moveLines;
            splitEndLine_ -= moveLines;
            return;
        }
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR sizeSplits, const SplitVTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    numSplit_ = tilingData_->gSize;
    int32_t splitBufSize = Ops::Base::CeilAlign(numSplit_ * SPLIT_UB_NUM * INT_NUM_TWO, ONE_BLOCK_UB);
    int32_t splitBufSizeInt32 = Ops::Base::CeilAlign(numSplit_ * SPLIT_UB_NUM, ONE_BLOCK_UB);
    pipe_.InitBuffer(splitBuf_, splitBufSize);
    pipe_.InitBuffer(splitBufInt32_, splitBufSizeInt32);
    if (tilingData_->isInt32 == 1) {
        sizeSplitsGmInt32_.SetGlobalBuffer((__gm__ int32_t *)sizeSplits);
        splitOffsetLocalInt32_ = splitBufInt32_.template Get<int32_t>();
    } else {
        sizeSplitsGm_.SetGlobalBuffer((__gm__ int64_t *)sizeSplits);
    }
    splitOffsetLocal_ = splitBuf_.template Get<int64_t>();
    int32_t ubSize = ((tilingData_->ubSize - splitBufSize - splitBufSizeInt32) / SPLIT_UB_NUM / ONE_BLOCK_UB) * ONE_BLOCK_UB;
    ubSizeNum_ = ubSize / sizeof(T);
    pipe_.InitBuffer(inQueueX_, bufferNum, ubSize);
    pipe_.InitBuffer(outQueueY_, bufferNum, ubSize);
    xGm.SetGlobalBuffer((__gm__ T *)x);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void *>(y));
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::DataCopyOutGm(int32_t blockCount, int64_t blockLen, int64_t stride,
    int64_t ubOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = stride * sizeof(T);
    DataCopyPad(yGm, yLocal_[ubOffset], copyParams);
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::OnceCopyOut(int32_t curNFactorBeginSplit, int64_t mOffset,
    int64_t nOffset, int64_t perSize, int32_t curNFactor, int32_t mUbFactorNow, LocalTensor<T> &xUb)
{
    int64_t curNiSize = CurSplitSize(curProcessSplit_);
    int64_t curSplitNSize = curNFactor;
    int64_t stride = curNiSize - curSplitNSize;
    int64_t startFirstOffset = (mBlockOffset_ + mOffset ) * curNiSize +
        (nBlockOffset_ + nOffset - perSize);
    yGm.SetGlobalBuffer(GetTensorAddr(curProcessSplit_, startFirstOffset));
    DataCopyExtParams copyParams;
    copyParams.blockCount = mUbFactorNow;
    copyParams.blockLen = curSplitNSize * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = stride * sizeof(T);
    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventID1);
    WaitFlag<HardEvent::MTE2_MTE3>(eventID1);
    DataCopyPad(yGm, xUb, copyParams);
    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    outQueueY_.FreeTensor(yLocal_);
    inQueueX_.FreeTensor(xUb);
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::DataCopyIn(int64_t mOffset, int64_t nOffset, int32_t curNFactor,
    int32_t mUbFactorNow, LocalTensor<T> &xUb)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = mUbFactorNow;
    copyParams.blockLen = curNFactor * sizeof(T);
    copyParams.srcStride = (tilingData_->nSize - curNFactor) * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPad(xUb,
        xGm[(mBlockOffset_ + mOffset) * tilingData_->nSize + nBlockOffset_ + nOffset],
        copyParams, padParams);
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::AllCopyOut(int64_t mOffset, int64_t nOffset, int32_t &curNFactor,
    int32_t mUbFactorNow, int64_t nowBlockEnd, LocalTensor<T> &xUb)
{
    splitEndLine_ = (nOffset + nBlockOffset_ + curNFactor) > nowBlockEnd ?
                    nowBlockEnd : (nOffset + nBlockOffset_ + curNFactor);
    int64_t splitStartOffset = nOffset + nBlockOffset_;
    int64_t preSize = 0;
    int64_t curSplitNSize = 0;
    int64_t curNFactorAlign = (curNFactor + oneUbNum_ - 1) / oneUbNum_ * oneUbNum_;
    int64_t curSplitNSizeAlign = 0;
    int64_t stride = 0;
    int64_t startFirstOffset = 0;
    int64_t curNiSize = CurSplitSize(curProcessSplit_);
    int64_t curNFactorOffset = 0;
    int64_t ubOffset = 0;
    int64_t mnAlignSize = 0;
    curNOffset_ = splitStartOffset;
    if (curProcessSplit_ - 1 >= 0) {
        preSize = curSplitOffset_ - curNiSize;
    }
    while (curNOffset_ < splitEndLine_) {
        curNiSize = CurSplitSize(curProcessSplit_);
        if (curSplitOffset_ - curNOffset_ > curNFactor && curNFactorOffset == 0) {
            // 当前ub处理不跨split块
            OnceCopyOut(curSplitOffset_, mOffset, nOffset, preSize, curNFactor, mUbFactorNow, xUb);
            curNOffset_ += curNFactor;
            return;
        } else if (curNOffset_ == splitStartOffset) {
            // 处理当前ub的第一个split块
            curSplitNSize = curSplitOffset_ - curNOffset_;
            curSplitNSizeAlign = ((curSplitNSize + oneUbNum_ - 1) / oneUbNum_) * oneUbNum_;
            stride = curNiSize - curSplitNSize;
            startFirstOffset = (mBlockOffset_ + mOffset) * curNiSize + stride;
            mnAlignSize = Ops::Base::CeilAlign(mUbFactorNow * curSplitNSizeAlign, BLOCK_ELENUM);
        } else if (splitEndLine_ > curSplitOffset_) {
            // 处理中间split块
            curSplitNSize = curNiSize;
            curSplitNSizeAlign = curSplitNSize;
            startFirstOffset = (mBlockOffset_ + mOffset) * curNiSize;
            mnAlignSize = Ops::Base::CeilAlign(mUbFactorNow * curSplitNSizeAlign, BLOCK_ELENUM);
            stride = 0;
        } else if (curSplitOffset_ >= splitEndLine_) {
            // 处理尾块
            ComputeEndLine(curNFactor, nOffset, nowBlockEnd);
            curSplitNSize = splitEndLine_ - curNOffset_;
            curSplitNSizeAlign = ((curSplitNSize + oneUbNum_ - 1) / oneUbNum_) * oneUbNum_;
            stride = curNiSize - curSplitNSize;
            startFirstOffset = (mBlockOffset_ + mOffset) * curNiSize;
            mnAlignSize = Ops::Base::CeilAlign(mUbFactorNow * curSplitNSizeAlign, BLOCK_ELENUM);
        }
        if (ubOffset + mnAlignSize > ubSizeNum_) {
            // 兜底分支，防止ub写越界
            splitEndLine_ = curNOffset_;
            curNFactor = splitEndLine_ - nOffset - nBlockOffset_;
            break;
        }
        if (curSplitNSizeAlign > maxGatherNum_) {
            DataCopyVf(mUbFactorNow, curSplitNSizeAlign, curNFactorOffset, curNFactorAlign, xUb, ubOffset);
        } else if (curSplitNSizeAlign <= 0) {
            curProcessSplit_ += 1;
            curSplitOffset_ += CurSplitSize(curProcessSplit_);
            if (curProcessSplit_ >= tilingData_->nBlockSplitOffsetEnd[blockIdx_]) {
                break;
            }
            continue;
        } else {
            DataCopyGatherVf(mUbFactorNow, curSplitNSizeAlign, curNFactorOffset, curNFactorAlign, xUb, ubOffset);
        }

        yGm.SetGlobalBuffer(GetTensorAddr(curProcessSplit_, startFirstOffset));
        event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID1);
        WaitFlag<HardEvent::V_MTE3>(eventID1);
        if (curNOffset_ == splitStartOffset || curSplitOffset_ >= splitEndLine_) {
            DataCopyOutGm(mUbFactorNow, curSplitNSize, stride, ubOffset);
        } else {
            DataCopyOutGm(1, curSplitNSize * mUbFactorNow, stride, ubOffset);
        }
        ubOffset += mnAlignSize;
        if (curSplitOffset_ > splitEndLine_) {
            curNOffset_ += curSplitNSize;
            curNFactorOffset = curNFactorOffset + curSplitNSize;
        } else {
            curNOffset_ += curSplitNSize;
            curNFactorOffset = curNFactorOffset + curSplitNSize;
            curProcessSplit_ += 1;
            curSplitOffset_ += CurSplitSize(curProcessSplit_);
        }
    }
    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    curNOffset_ = splitEndLine_;
    outQueueY_.FreeTensor(yLocal_);
    inQueueX_.FreeTensor(xUb);
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplit<T, U, Y>::UbProcess(int32_t mTimes, int32_t mUbFactor, int64_t nBlockFactorNow,
    int64_t mBlockFactorNow, int32_t nUbFactor, int64_t nowBlockEnd)
{
    for (int32_t m = 0; m < mTimes; m++) {
        int32_t mUbFactorNow = mUbFactor;
        if (m == mTimes - 1) {
            mUbFactorNow = mBlockFactorNow - mUbFactor * (mTimes - 1);
        }
        int32_t curNFactorBeginSplit = tilingData_->nBlockSplitOffset[blockIdx_];
        int32_t curNFactorEndSplit = tilingData_->nBlockSplitOffset[blockIdx_];
        int32_t curNFactor = 0;
        int32_t endLinePosition = curNFactorBeginSplit;
        curProcessSplit_ = tilingData_->nBlockSplitOffset[blockIdx_];
        splitEndLine_ = nBlockOffset_;
        curNOffset_ = nBlockOffset_;
        curSplitOffset_ = splitPrefixNow_;

        for (int64_t nOffset = 0; nOffset < nBlockFactorNow; nOffset += curNFactor) {
            curNFactorBeginSplit = endLinePosition;
            curNFactor = nUbFactor;
            if (nBlockOffset_ + nOffset + curNFactor > tilingData_->nSize) {
                curNFactor = tilingData_->nSize - nBlockOffset_ - nOffset;
            }
            LocalTensor<T> xUb = inQueueX_.AllocTensor<T>();
            yLocal_ = outQueueY_.AllocTensor<T>();
            int64_t mOffset = m *mUbFactor;
            DataCopyIn(mOffset, nOffset, curNFactor, mUbFactorNow, xUb);
            inQueueX_.EnQue(xUb);
            LocalTensor<T> xUbSize = inQueueX_.DeQue<T>();
            AllCopyOut(mOffset, nOffset, curNFactor, mUbFactorNow, nowBlockEnd, xUbSize);
        }
    }
}

template <typename T, typename U, typename Y> __aicore__ inline void SplitVUbSplit<T, U, Y>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }
    if (tilingData_->isInt32 == 1) {
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<int32_t> padParamsIdx = {false, 0, 0, 0};
        copyParams.blockCount = 1;
        copyParams.blockLen = numSplit_ * sizeof(int32_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(splitOffsetLocalInt32_, sizeSplitsGmInt32_, copyParams, padParamsIdx);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
        AscendC::Cast(splitOffsetLocal_, splitOffsetLocalInt32_, RoundMode::CAST_NONE, numSplit_);
        AscendC::Muls(
            splitOffsetLocal_, splitOffsetLocal_, static_cast<int32_t>(tilingData_->sizeAfterSplitDim), numSplit_);
    } else {
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<int64_t> padParamsIdx = {false, 0, 0, 0};
        copyParams.blockCount = 1;
        copyParams.blockLen = numSplit_ * sizeof(int64_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(splitOffsetLocal_, sizeSplitsGm_, copyParams, padParamsIdx);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
        AscendC::Muls(splitOffsetLocal_, splitOffsetLocal_, tilingData_->sizeAfterSplitDim, numSplit_);
    }
    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID1);
    WaitFlag<HardEvent::V_S>(eventID1);
    if (tilingData_->negIdx != -1) {
        uint32_t index = tilingData_->negIdx;
        int64_t value = tilingData_->negValue * tilingData_->sizeAfterSplitDim;
        splitOffsetLocal_.SetValue(index, value);
    }
    int32_t mblock = blockIdx_ / tilingData_->nBlockCount;
    int32_t nblock = blockIdx_ % tilingData_->nBlockCount;
    splitPrefixNow_ = tilingData_->nBlockSplitPrefixStart[blockIdx_];
    splitPrefixNext_ = tilingData_->nBlockSplitPrefixEnd[blockIdx_];

    if (nblock <= tilingData_->nBlockFactorNum) {
        nBlockOffset_ = nblock * tilingData_->nBlockFactor;
    } else {
        nBlockOffset_ = tilingData_->nBlockFactorNum * tilingData_->nBlockFactor +
            (nblock - tilingData_->nBlockFactorNum) * tilingData_->nBlockFactorTail;
    }

    if (mblock <= tilingData_->mBlockFactorNum) {
        mBlockOffset_ = mblock * tilingData_->mBlockFactor;
    } else {
        mBlockOffset_ = tilingData_->mBlockFactorNum * tilingData_->mBlockFactor +
            (mblock - tilingData_->mBlockFactorNum) * tilingData_->mBlockFactorTail;
    }
    int32_t mUbFactor = tilingData_->mBlockFactor > oneUbNum_ ? oneUbNum_ : tilingData_->mBlockFactor;
    if (tilingData_->gSize > HUGE_NUM_SPLIT && tilingData_->mSize >= MORE_M) {
        mUbFactor = MORE_M;
    }
    int32_t mUbSize = ubSizeNum_ / mUbFactor / oneUbNum_ * oneUbNum_;
    int32_t nUbFactor = mUbSize;
    if ((nblock >= tilingData_->nBlockFactorNum) && tilingData_->nBlockFactorAlign <= nUbFactor) {
        nUbFactor = tilingData_->nBlockFactorTail;
    } else if (tilingData_->nBlockFactorAlign <= nUbFactor) {
        nUbFactor = tilingData_->nBlockFactor;
    }

    int64_t mBlockFactorNow = tilingData_->mBlockFactor;
    if (mblock >= tilingData_->mBlockFactorNum) {
        mBlockFactorNow = tilingData_->mBlockFactorTail;
    }
    int64_t mUbFactorAlign = (nUbFactor + oneUbNum_ - 1) / oneUbNum_ * oneUbNum_ + INT_NUM_TWO * oneUbNum_;
    int64_t lastUbNum = (ubSizeNum_ - mUbFactorAlign * mUbFactor) / mUbFactorAlign;
    if (tilingData_->mBlockFactor <= (lastUbNum + mUbFactor)) {
        mUbFactor = tilingData_->mBlockFactor;
    } else {
        mUbFactor = mUbFactor + (lastUbNum / oneUbNum_) * oneUbNum_;
    }
    int32_t mTimes = (mBlockFactorNow + mUbFactor - 1) / mUbFactor;
    int64_t nBlockFactorNow = tilingData_->nBlockFactor;
    if (nblock >= tilingData_->nBlockFactorNum) {
        nBlockFactorNow = tilingData_->nBlockFactorTail;
    }
    int64_t nowBlockEnd = nBlockOffset_ + nBlockFactorNow;
    UbProcess(mTimes, mUbFactor, nBlockFactorNow, mBlockFactorNow, nUbFactor, nowBlockEnd);
}
}
#endif // namespace SplitV
