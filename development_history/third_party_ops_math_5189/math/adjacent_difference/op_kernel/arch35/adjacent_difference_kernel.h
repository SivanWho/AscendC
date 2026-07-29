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
 * \file adjacent_difference_kernel.h
 * \brief
 */

#ifndef ADJACENT_DIFFERENCE_KERNEL_H
#define ADJACENT_DIFFERENCE_KERNEL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t DOUBLE_INT32 = 2;
constexpr uint32_t BLOCKSIZE = Ops::Base::GetUbBlockSize();

template <typename EQS_TYPE, typename INPUT_TYPE, typename OUT_TYPE>
class AdjacentDifferenceKernel
{
public:
    __aicore__ inline AdjacentDifferenceKernel(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AdjacentDifferenceTilingData* tilingData)
    {
        totalSize_ = tilingData->totalSize;
        tilingNum_ = tilingData->tilingNum;
        coreNum_ = tilingData->coreNum;
        vLength_ = Ops::Base::GetVRegSize() / sizeof(EQS_TYPE);

        x1Gm_.SetGlobalBuffer((__gm__ EQS_TYPE*)(x));
        yGm_.SetGlobalBuffer((__gm__ int32_t*)(y));

        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, tilingNum_ * sizeof(EQS_TYPE));
        
        if constexpr (sizeof(OUT_TYPE) == sizeof(int64_t)) {
            uint32_t bufferSize = tilingNum_ * sizeof(OUT_TYPE) + 2 * BLOCKSIZE;
            pipe_->InitBuffer(yQueue_, BUFFER_NUM, bufferSize);
            pipe_->InitBuffer(onesBuf_, sizeof(OUT_TYPE));
        } else if constexpr (sizeof(OUT_TYPE) == sizeof(int32_t)) {
            pipe_->InitBuffer(yQueue_, BUFFER_NUM, tilingNum_ * sizeof(OUT_TYPE));
            pipe_->InitBuffer(onesBuf_, sizeof(OUT_TYPE));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t BlockId = GetBlockIdx();
        int64_t startIdx = BlockId * (tilingNum_ - 1);
        if  (startIdx > totalSize_) {
            return;
        }
        uint32_t loop = 0;
        for (int64_t remainLen = totalSize_ - BlockId * (tilingNum_ - 1), idx = 0; remainLen - 1 > 0; remainLen -= coreNum_ * (tilingNum_ - 1), idx++) {
            uint32_t copyLen = min(tilingNum_, totalSize_ - startIdx);
            GetResult(startIdx, copyLen);
            startIdx += coreNum_ * (tilingNum_ - 1);
        }

        if (GetBlockIdx() == 0 && totalSize_ > 0) {
            CopyOutYFirst();
        }
    }

    __aicore__ inline void CopyInX(int64_t startIdx, uint32_t copyLen) 
    {
        DataCopyPadExtParams<EQS_TYPE> padParams;
        padParams.isPad = false;
        padParams.paddingValue = 0;
        padParams.rightPadding = 0;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(EQS_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        LocalTensor<EQS_TYPE> x1Local = x1Queue_.template AllocTensor<EQS_TYPE>();
        DataCopyPad(x1Local, x1Gm_[startIdx], dataCopyParams, padParams);
        x1Queue_.EnQue(x1Local);
    }

    __aicore__ inline void CopyOutY(int64_t startIdx, uint32_t copyLen) 
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(OUT_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        if constexpr (sizeof(OUT_TYPE) == sizeof(int64_t)) {
            LocalTensor<int32_t> yLocal = yQueue_.template DeQue<int32_t>();
            DataCopyPad(yGm_[DOUBLE_INT32 * (startIdx + 1)], yLocal, dataCopyParams);
            yQueue_.FreeTensor(yLocal);
        } else if constexpr (sizeof(OUT_TYPE) == sizeof(int32_t)) {
            LocalTensor<int32_t> yLocal = yQueue_.template DeQue<int32_t>();
            DataCopyPad(yGm_[startIdx + 1], yLocal, dataCopyParams);
            yQueue_.FreeTensor(yLocal);
        }
    }

    __aicore__ inline void CopyOutYFirst() 
    {
        LocalTensor<int32_t> onesTensor = onesBuf_.Get<int32_t>();
        Duplicate(onesTensor, (int32_t)0, sizeof(OUT_TYPE)/sizeof(int32_t));
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(OUT_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopyPad(yGm_[0], onesTensor, dataCopyParams);
    }

    __aicore__ inline void adjacentDifference64(LocalTensor<EQS_TYPE>& x1Local, LocalTensor<int32_t>& outTensor, uint32_t copyLen, uint16_t& nLoop, uint32_t& alignPosition) 
    {
        __local_mem__ EQS_TYPE* sourceX1 = (__ubuf__ EQS_TYPE*)x1Local.GetPhyAddr();
        __local_mem__ int32_t* dstAddr = sizeof(OUT_TYPE) == sizeof(int64_t) ? (__ubuf__ int32_t*)outTensor.GetPhyAddr(alignPosition)\
                                                                             : (__ubuf__ int32_t*)outTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t inputElementNum = copyLen;
            AscendC::MicroAPI::RegTensor<int32_t> regZero, regOne;
            MicroAPI::MaskReg maskRegInt32 = MicroAPI::CreateMask<int32_t>();
            MicroAPI::Duplicate(regZero, 0, maskRegInt32);
            MicroAPI::Duplicate(regOne, 1, maskRegInt32);
            for (uint16_t i = 0; i < (uint16_t)nLoop; i++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<EQS_TYPE>(inputElementNum);
                AscendC::MicroAPI::RegTensor<EQS_TYPE> regX1, regX2;
                AscendC::MicroAPI::RegTensor<int32_t> regY;
                AscendC::MicroAPI::UnalignReg u0;
                AscendC::MicroAPI::MaskReg cmpMaskReg, lowerCmpMaskReg, lowerMaskReg;
                DataCopy(regX1, sourceX1 + i * vLength_);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, sourceX1 + 1 + i * vLength_);
                AscendC::MicroAPI::DataCopyUnAlign(regX2, u0, sourceX1 + 1 + i * vLength_);

                AscendC::MicroAPI::Compare<INPUT_TYPE, CMPMODE::NE>(cmpMaskReg, (MicroAPI::RegTensor<INPUT_TYPE>&)regX1, (MicroAPI::RegTensor<INPUT_TYPE>&)regX2, mask);
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(lowerCmpMaskReg, cmpMaskReg);

                AscendC::MicroAPI::Select(regY, regOne, regZero, lowerCmpMaskReg);

                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(lowerMaskReg, mask);
                AscendC::MicroAPI::DataCopy(dstAddr + i * vLength_, regY, lowerMaskReg);
            }
        }
    }

    __aicore__ inline void adjacentDifference32(LocalTensor<EQS_TYPE>& x1Local, LocalTensor<int32_t>& outTensor, uint32_t copyLen, uint16_t& nLoop, uint32_t& alignPosition) 
    {
        __local_mem__ EQS_TYPE* sourceX1 = (__ubuf__ EQS_TYPE*)x1Local.GetPhyAddr();
        __local_mem__ int32_t* dstAddr = sizeof(OUT_TYPE) == sizeof(int64_t) ? (__ubuf__ int32_t*)outTensor.GetPhyAddr(alignPosition)\
                                                                             : (__ubuf__ int32_t*)outTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int32_t> regZero, regOne;
            MicroAPI::MaskReg maskRegInt32 = MicroAPI::CreateMask<int32_t>();
            MicroAPI::Duplicate(regZero, 0, maskRegInt32);
            MicroAPI::Duplicate(regOne, 1, maskRegInt32);
            uint32_t inputElementNum = copyLen;
            for (uint16_t i = 0; i < (uint16_t)nLoop; i++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<EQS_TYPE>(inputElementNum);
                AscendC::MicroAPI::RegTensor<EQS_TYPE> regX1, regX2;
                AscendC::MicroAPI::RegTensor<int32_t> regY;
                AscendC::MicroAPI::UnalignReg u0;
                AscendC::MicroAPI::MaskReg cmpMaskReg;
                DataCopy(regX1, sourceX1 + i * vLength_);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, sourceX1 + 1 + i * vLength_);
                AscendC::MicroAPI::DataCopyUnAlign(regX2, u0, sourceX1 + 1 + i * vLength_);

                AscendC::MicroAPI::Compare<INPUT_TYPE, CMPMODE::NE>(cmpMaskReg, (MicroAPI::RegTensor<INPUT_TYPE>&)regX1, (MicroAPI::RegTensor<INPUT_TYPE>&)regX2, mask);
                AscendC::MicroAPI::Select(regY, regOne, regZero, cmpMaskReg);
                AscendC::MicroAPI::DataCopy(dstAddr + i * vLength_, regY, mask);
            }
        }
    }

    __aicore__ inline void adjacentDifference16(LocalTensor<EQS_TYPE>& x1Local, LocalTensor<int32_t>& outTensor, uint32_t copyLen, uint16_t& nLoop, uint32_t& alignPosition) 
    {
        __local_mem__ EQS_TYPE* sourceX1 = (__ubuf__ EQS_TYPE*)x1Local.GetPhyAddr();
        __local_mem__ int32_t* dstAddr = sizeof(OUT_TYPE) == sizeof(int64_t) ? (__ubuf__ int32_t*)outTensor.GetPhyAddr(alignPosition)\
                                                                             : (__ubuf__ int32_t*)outTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t inputElementNum = copyLen;
            MicroAPI::MaskReg maskRegInt32 = MicroAPI::CreateMask<int32_t>();
            AscendC::MicroAPI::RegTensor<int32_t> regZero, regOne;
            MicroAPI::Duplicate(regZero, 0, maskRegInt32);
            MicroAPI::Duplicate(regOne, 1, maskRegInt32);
            for (uint16_t i = 0; i < (uint16_t)nLoop; i++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<EQS_TYPE>(inputElementNum);
                AscendC::MicroAPI::RegTensor<EQS_TYPE> regX1, regX2;
                AscendC::MicroAPI::RegTensor<int32_t> lowerRegY, higherRegY;
                AscendC::MicroAPI::UnalignReg u0;
                AscendC::MicroAPI::MaskReg cmpMaskReg, lowerCmpMaskReg, highCmpMaskReg, lowerMaskReg, higherMaskReg;
                DataCopy(regX1, sourceX1 + i * vLength_);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, sourceX1 + 1 + i * vLength_);
                AscendC::MicroAPI::DataCopyUnAlign(regX2, u0, sourceX1 + 1 + i * vLength_);

                AscendC::MicroAPI::Compare<INPUT_TYPE, CMPMODE::NE>(cmpMaskReg, (MicroAPI::RegTensor<INPUT_TYPE>&)regX1, (MicroAPI::RegTensor<INPUT_TYPE>&)regX2, mask);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(lowerCmpMaskReg, cmpMaskReg);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(highCmpMaskReg, cmpMaskReg);

                AscendC::MicroAPI::Select(lowerRegY, regOne, regZero, lowerCmpMaskReg);
                AscendC::MicroAPI::Select(higherRegY, regOne, regZero, highCmpMaskReg);

                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(lowerMaskReg, mask);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(higherMaskReg, mask);
                AscendC::MicroAPI::DataCopy(dstAddr + i * vLength_, lowerRegY, lowerMaskReg);
                AscendC::MicroAPI::DataCopy(dstAddr + i * vLength_ + vLength_ / 2, higherRegY, higherMaskReg);
            }
        }
    }

    __aicore__ inline void adjacentDifference8(LocalTensor<EQS_TYPE>& x1Local, LocalTensor<int32_t>& outTensor, uint32_t copyLen, uint16_t& nLoop, uint32_t& alignPosition)
    {
        __local_mem__ EQS_TYPE* sourceX1 = (__ubuf__ EQS_TYPE*)x1Local.GetPhyAddr();
        __local_mem__ int32_t* dstAddr = sizeof(OUT_TYPE) == sizeof(int64_t) ? (__ubuf__ int32_t*)outTensor.GetPhyAddr(alignPosition)\
                                                                             : (__ubuf__ int32_t*)outTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg maskRegInt32 = MicroAPI::CreateMask<int32_t>();
            AscendC::MicroAPI::RegTensor<int32_t> regZero, regOne;
            MicroAPI::Duplicate(regZero, 0, maskRegInt32);
            MicroAPI::Duplicate(regOne, 1, maskRegInt32);
            uint32_t inputElementNum = copyLen;
            for (uint16_t i = 0; i < (uint16_t)nLoop; i++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<EQS_TYPE>(inputElementNum);
                AscendC::MicroAPI::RegTensor<EQS_TYPE> regX1, regX2;
                AscendC::MicroAPI::RegTensor<int32_t> lowerLowerRegY, lowerHigherRegY, highLowerRegY, highHigherRegY;
                AscendC::MicroAPI::UnalignReg u0;
                AscendC::MicroAPI::MaskReg cmpMaskReg, 
                                    lowerCmpMaskReg, lowerLowerCmpMaskReg, lowerHighCmpMaskReg,
                                    highCmpMaskReg, highLowerCmpMaskReg, highHighCmpMaskReg,
                                    lowerMaskReg, lowerLowerMaskReg, lowerHighMaskReg,
                                    higherMaskReg, highLowerMaskReg, highHighMaskReg;
                DataCopy(regX1, sourceX1 + i * vLength_);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, sourceX1 + 1 + i * vLength_);
                AscendC::MicroAPI::DataCopyUnAlign(regX2, u0, sourceX1 + 1 + i * vLength_);

                AscendC::MicroAPI::Compare<INPUT_TYPE, CMPMODE::NE>(cmpMaskReg, (MicroAPI::RegTensor<INPUT_TYPE>&)regX1, (MicroAPI::RegTensor<INPUT_TYPE>&)regX2, mask);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(lowerCmpMaskReg, cmpMaskReg);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(lowerLowerCmpMaskReg, lowerCmpMaskReg);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(lowerHighCmpMaskReg, lowerCmpMaskReg);

                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(highCmpMaskReg, cmpMaskReg);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(highLowerCmpMaskReg, highCmpMaskReg);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(highHighCmpMaskReg, highCmpMaskReg);

                AscendC::MicroAPI::Select(lowerLowerRegY, regOne, regZero, lowerLowerCmpMaskReg);
                AscendC::MicroAPI::Select(lowerHigherRegY, regOne, regZero, lowerHighCmpMaskReg);
                AscendC::MicroAPI::Select(highLowerRegY, regOne, regZero, highLowerCmpMaskReg);
                AscendC::MicroAPI::Select(highHigherRegY, regOne, regZero, highHighCmpMaskReg);

                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(lowerMaskReg, mask);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(lowerLowerMaskReg, lowerMaskReg);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(lowerHighMaskReg, lowerMaskReg);

                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(higherMaskReg, mask);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(highLowerMaskReg, higherMaskReg);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(highHighMaskReg, higherMaskReg);

                AscendC::MicroAPI::DataCopy(dstAddr + i * vLength_, lowerLowerRegY, lowerLowerMaskReg);
                AscendC::MicroAPI::DataCopy(dstAddr + i * vLength_ + vLength_ / 4, lowerHigherRegY, lowerHighMaskReg);
                AscendC::MicroAPI::DataCopy(dstAddr + i * vLength_ + 2 * vLength_ / 4, highLowerRegY, highLowerMaskReg);
                AscendC::MicroAPI::DataCopy(dstAddr + i * vLength_ + 3 * vLength_ / 4, highHigherRegY, highHighMaskReg);
            }
        }
    }

    __aicore__ inline void compute(int64_t startIdx, uint32_t copyLen)
    {
        LocalTensor<EQS_TYPE> x1Local = x1Queue_.template DeQue<EQS_TYPE>();
        LocalTensor<int32_t> yLocal = yQueue_.template AllocTensor<int32_t>();
        uint32_t alignPosition = (copyLen * sizeof(int32_t) + BLOCKSIZE - 1) / BLOCKSIZE * BLOCKSIZE / sizeof(int32_t);
        uint16_t nLoop = (copyLen + vLength_ - 1) / vLength_;
        if constexpr (sizeof(EQS_TYPE) == sizeof(int64_t)) {
            adjacentDifference64(x1Local, yLocal, copyLen, nLoop, alignPosition);
        } else if constexpr (sizeof(EQS_TYPE) == sizeof(int32_t)) {
            adjacentDifference32(x1Local, yLocal, copyLen, nLoop, alignPosition);
        } else if constexpr (sizeof(EQS_TYPE) == sizeof(int16_t)) {
            adjacentDifference16(x1Local, yLocal, copyLen, nLoop, alignPosition);
        } else if constexpr (sizeof(EQS_TYPE) == sizeof(int8_t)) {
            adjacentDifference8(x1Local, yLocal, copyLen, nLoop, alignPosition);
        }
        PipeBarrier<PIPE_V>();
        x1Queue_.FreeTensor(x1Local);
        if constexpr (sizeof(OUT_TYPE) == sizeof(int64_t)) {
            LocalTensor<int64_t> dstLocal = yLocal.template ReinterpretCast<int64_t>();
            Cast(dstLocal, yLocal[alignPosition], RoundMode::CAST_NONE, copyLen - 1);
        }
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void GetResult(int64_t startIdx, uint32_t copyLen)
    {
        CopyInX(startIdx, copyLen);
        compute(startIdx, copyLen);
        CopyOutY(startIdx, copyLen - 1);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TBuf<TPosition::VECCALC> onesBuf_;

    GlobalTensor<EQS_TYPE> x1Gm_;
    GlobalTensor<int32_t> yGm_;

    int64_t totalSize_;
    int64_t coreNum_;
    int64_t tilingNum_;
    uint32_t vLength_;

    TPipe* pipe_ = nullptr;
};

#endif  // ADJACENT_DIFFERENCE_KERNEL_H