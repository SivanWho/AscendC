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
 * \file chunk_cat_arch35.h
 * \brief
 */

#ifndef _CHUNK_CAT_ARCH35_H_
#define _CHUNK_CAT_ARCH35_H_

#include "chunk_cat_common.h"
#include "op_kernel/platform_util.h"

using namespace AscendC;
using namespace Ops::Base;

static constexpr MicroAPI::CastTrait castTraitZero = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN
};
static constexpr MicroAPI::CastTrait castTraitOne = {
    MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN
};
static constexpr MicroAPI::CastTrait castTraitbf2half = {
    MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT
};
static constexpr MicroAPI::CastTrait castTraithalf2bf = {
    MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT
};

template <typename T1, typename T2, bool NEED_CAST = false>
class ChunkCatArch35 : public ChunkCatCommon<T1, T2>
{
public:
    __aicore__ inline ChunkCatArch35(TPipe *pipe) : ChunkCatCommon<T1, T2>(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ChunkCatTilingData& tilingData)
    {
        this->InitCommon(x, y, tilingData);
    }

    __aicore__ inline void Process()
    {
        int64_t rowLoop = this->GetAlign(this->currentBlockRowFactor_, this->ubRowFactor_) / this->ubRowFactor_;
        int64_t colLoop = this->GetAlign(this->currentBlockColFactor_, this->ubColFactor_) / this->ubColFactor_;
        int64_t rowTail = this->currentBlockRowFactor_ % this->ubRowFactor_;
        int64_t colTail = this->currentBlockColFactor_ % this->ubColFactor_;

        uint64_t buf[10];
        this->desc_.SetShapeAddr(buf); // 用于获取shape信息
        
        for (int64_t i = 0; i < rowLoop * colLoop; i++) {
            UbLoopInfo ubLoopInfo{};
            ubLoopInfo.ubRowGroup = i / colLoop;
            ubLoopInfo.ubColGroup = i % colLoop;
            ubLoopInfo.currentUbRowFactor = (rowTail != 0 && ubLoopInfo.ubRowGroup == rowLoop - 1) ?
                                            rowTail : this->ubRowFactor_;
            ubLoopInfo.currentUbColFactor = (colTail != 0 && ubLoopInfo.ubColGroup == colLoop - 1) ?
                                            colTail : this->ubColFactor_;
            // 搬入&&计算
            CopyInAndCompute(ubLoopInfo);
            SetFlag<HardEvent::V_MTE3>(this->event_);
            WaitFlag<HardEvent::V_MTE3>(this->event_);
            // 搬出
            CopyOut(ubLoopInfo);
        }
    }

private:
    __aicore__ inline void CopyInAndCompute(UbLoopInfo& ubLoopInfo)
    {
        int64_t localOffset = 0;
        int64_t totalCol = 0;
        ubLoopInfo.colStart = this->blockColGroup_ * this->blockColFactor_ + ubLoopInfo.ubColGroup * this->ubColFactor_;
        ubLoopInfo.rowStart = this->blockRowGroup_ * this->blockRowFactor_ + ubLoopInfo.ubRowGroup * this->ubRowFactor_;
        for (uint32_t i = 0; i < this->inputNum_; i++) {
            if (ubLoopInfo.totalUbCol >= ubLoopInfo.currentUbColFactor) {
                break;
            }
            TensorInfo tensorInfo{};
            this->srcGlobal_.SetGlobalBuffer(this->inputList_.template GetDataPtr<T1>(i));
            this->GetChunkInfo(i, tensorInfo);
            if (!this->IsTensorInRange(totalCol, ubLoopInfo, tensorInfo)) {
                totalCol += tensorInfo.tensorCol;
            } else {
                tensorInfo.chunkRow = tensorInfo.chunkDimSize / tensorInfo.chunkCol;
                tensorInfo.chunkRowAlign = this->GetAlign(tensorInfo.chunkDimSize, tensorInfo.chunkCol) / tensorInfo.chunkCol;
                this->SplitTensorDim0(totalCol, ubLoopInfo, tensorInfo);
                if (ubLoopInfo.rowStart >= tensorInfo.chunkRowAlign) {
                    tensorInfo.isZero = true;
                } else {
                    this->CopyInChunk(totalCol, localOffset, ubLoopInfo, tensorInfo);
                }
                // 计算
                Compute(ubLoopInfo, tensorInfo, totalCol, localOffset);
                ubLoopInfo.count++;
                ubLoopInfo.totalUbCol += tensorInfo.splitCol;
                totalCol += tensorInfo.tensorCol;
                localOffset += this->GetAlign(ubLoopInfo.currentUbRowFactor * tensorInfo.splitCol, this->srcEleUbBlock_);
            }

            if (ubLoopInfo.count > 31) {
                // 32个tensor处理一次
                CopyOut(ubLoopInfo);
                localOffset = 0;
                ubLoopInfo.preCatCol += ubLoopInfo.totalUbCol;
                ubLoopInfo.totalUbCol = 0;
            }
        }
    }

    __aicore__ inline void Compute(const UbLoopInfo& ubLoopInfo, const TensorInfo& tensorInfo, int64_t totalCol, int64_t localOffset)
    {
        if (!tensorInfo.isZero) {
            SetFlag<HardEvent::MTE2_V>(this->event_);
            WaitFlag<HardEvent::MTE2_V>(this->event_);
        }
        CopyCatVF(ubLoopInfo, tensorInfo, totalCol, localOffset);
        SetFlag<HardEvent::V_MTE3>(this->event_);
        WaitFlag<HardEvent::V_MTE3>(this->event_);
    }

    __aicore__ inline void CopyOut(const UbLoopInfo& ubLoopInfo)
    {
        uint16_t blockCount = ubLoopInfo.currentUbRowFactor;
        uint32_t blockLen = ubLoopInfo.totalUbCol * sizeof(T2);
        uint32_t dstStride = (this->outputCol_ - ubLoopInfo.totalUbCol)* sizeof(T2);
        uint32_t srcStride = (this->GetAlign(ubLoopInfo.currentUbColFactor, this->dstEleUbBlock_) -
                              this->GetAlign(ubLoopInfo.totalUbCol, this->dstEleUbBlock_)) / this->dstEleUbBlock_;
        DataCopyExtParams copyParamsOut{blockCount, blockLen, srcStride, dstStride, 0};
        int64_t dstOffset = ubLoopInfo.ubRowGroup * this->ubRowFactor_ * this->outputCol_ +
                            ubLoopInfo.ubColGroup * this->ubColFactor_ + ubLoopInfo.preCatCol;

        DataCopyPad(this->dstGlobal_[dstOffset], this->dstLocal_, copyParamsOut);
        SetFlag<HardEvent::MTE3_MTE2>(this->event_);
        WaitFlag<HardEvent::MTE3_MTE2>(this->event_);
        SetFlag<HardEvent::MTE3_V>(this->event_);
        WaitFlag<HardEvent::MTE3_V>(this->event_);
    }
    // Compute
    __aicore__ inline void DoCopyCatVF(__ubuf__ T2* dstAddr, __ubuf__ T1* srcAddr, uint16_t rowLoop, uint16_t colLoop,
                                        uint32_t tail, uint32_t rowStride)
    {
        uint32_t main = GetVRegSize() / sizeof(T1);
        uint32_t mainFP32 = GetVRegSize() / sizeof(T2);
        uint32_t tailFP32Fir = tail > mainFP32 ? mainFP32 : tail;
        uint32_t tailFP32Sec = tail - tailFP32Fir;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg uReg;
        AscendC::MicroAPI::RegTensor<T1> srcReg0;
        AscendC::MicroAPI::RegTensor<T2> dstReg0;
        AscendC::MicroAPI::RegTensor<T2> dstReg1;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T1, AscendC::MicroAPI:: MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
        for (uint16_t i = 0; i < rowLoop; i++) {
            auto curDstAddr = dstAddr + i * rowStride;
            for (uint16_t j = 0; j < colLoop; j++) {
                AscendC::MicroAPI::DataCopyUnAlign(srcReg0, u0, srcAddr, main);
                if constexpr (std::is_same_v<T1, T2>) {
                    AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, srcReg0, uReg, main);
                } else if constexpr (std::is_same_v<T1, half> && std::is_same_v<T2, bfloat16_t>) {
                    AscendC::MicroAPI::Cast<T2, T1, castTraithalf2bf>(dstReg0, srcReg0, mask);
                    AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, main);
                } else if constexpr (std::is_same_v<T1, bfloat16_t> && std::is_same_v<T2, half>) {
                    AscendC::MicroAPI::Cast<T2, T1, castTraitbf2half>(dstReg0, srcReg0, mask);
                    AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, main);
                } else {
                    AscendC::MicroAPI::Cast<T2, T1, castTraitZero>(dstReg0, srcReg0, mask);
                    AscendC::MicroAPI::Cast<T2, T1, castTraitOne>(dstReg1, srcReg0, mask);
                    AscendC::MicroAPI::Interleave(dstReg0, dstReg1, dstReg0, dstReg1);
                    AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, mainFP32);
                    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_STORE>();
                    AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg1, uReg, mainFP32);
                }
            }
            AscendC::MicroAPI::DataCopyUnAlign(srcReg0, u0, srcAddr, tail);
            if constexpr (std::is_same_v<T1, T2>) {
                AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, srcReg0, uReg, tail);
            } else if constexpr (std::is_same_v<T1, half> && std::is_same_v<T2, bfloat16_t>) {
                AscendC::MicroAPI::Cast<T2, T1, castTraithalf2bf>(dstReg0, srcReg0, mask);
                AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, tail);
            } else if constexpr (std::is_same_v<T1, bfloat16_t> && std::is_same_v<T2, half>) {
                AscendC::MicroAPI::Cast<T2, T1, castTraitbf2half>(dstReg0, srcReg0, mask);
                AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, tail);
            } else {
                AscendC::MicroAPI::Cast<T2, T1, castTraitZero>(dstReg0, srcReg0, mask);
                AscendC::MicroAPI::Cast<T2, T1, castTraitOne>(dstReg1, srcReg0, mask);
                AscendC::MicroAPI::Interleave(dstReg0, dstReg1, dstReg0, dstReg1);
                AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, tailFP32Fir);
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_STORE>();
                AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg1, uReg, tailFP32Sec);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(curDstAddr, uReg, 0);
        }
    }

    __aicore__ inline void DoPadCatVF(__ubuf__ T2* dstAddr, uint16_t rowLoop, uint16_t colLoop,
                                        uint32_t main, uint32_t tail, uint32_t rowStride)
    {
        AscendC::MicroAPI::UnalignReg uReg;
        AscendC::MicroAPI::RegTensor<T2> dstReg0;

        // // 纯pad
        T2 scalarValue = 0;
        AscendC::MicroAPI::Duplicate(dstReg0, scalarValue);
        for (uint16_t i = 0; i < rowLoop; i++) {
            auto curDstAddr = dstAddr + i * rowStride;
            for (uint16_t j = 0; j < colLoop; j++) {
                AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, main);
            }
            AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, tail);
            AscendC::MicroAPI::DataCopyUnAlignPost(curDstAddr, uReg, 0);
        }
    }

    __aicore__ inline void CopyCatVF(const UbLoopInfo& ubLoopInfo, const TensorInfo& tensorInfo, int64_t totalCol, int64_t localOffset)
    {
        uint32_t srcLen = tensorInfo.splitCol;
        bool isSplit = tensorInfo.isSplit;
        
        // 无搬运&&完整搬运阶段参数
        uint32_t main = GetVRegSize() / sizeof(T1);
        uint32_t mainFP32 = GetVRegSize() / sizeof(T2);
        uint16_t colLoop = srcLen / main; // 一行需要几次循环
        uint32_t tail = srcLen - colLoop * main;
        // fp16/bf16 -> fp32
        uint16_t colLoopFP32 = srcLen / mainFP32; // 一行需要几次循环
        uint32_t tailFP32 = srcLen - colLoopFP32 * mainFP32;
        uint32_t tailFP32Fir = tail > mainFP32 ? mainFP32 : tail;
        uint32_t tailFP32Sec = tail - tailFP32Fir;
        uint32_t rowStride = this->GetAlign(ubLoopInfo.currentUbColFactor, this->dstEleUbBlock_);

        // 三个部分的行数
        uint16_t rowLoop0 = 0;
        uint16_t rowLoop1 = 0;
        uint16_t rowLoop2 = 0;
        // 部分pad情况下pad部分
        uint16_t colLen0 = 0;
        uint16_t colLen1 = 0;
        uint16_t colLoop1 = 0;
        uint32_t tailPad = 0;
        
        if (ubLoopInfo.rowStart >= tensorInfo.chunkRowAlign) {
            // 无搬运，纯pad
            rowLoop2 = ubLoopInfo.currentUbRowFactor;
        } else if (ubLoopInfo.rowStart + ubLoopInfo.currentUbRowFactor < tensorInfo.chunkRowAlign) {
            // 完整搬运，无需pad
            rowLoop0 = ubLoopInfo.currentUbRowFactor;
        } else {
            // 部分搬运，需要pad
            rowLoop0 = tensorInfo.chunkRow - ubLoopInfo.rowStart;
            rowLoop1 = tensorInfo.chunkRowAlign - tensorInfo.chunkRow;
            rowLoop2 = ubLoopInfo.rowStart + ubLoopInfo.currentUbRowFactor - tensorInfo.chunkRowAlign;
            if (rowLoop1 > 0) {
                uint32_t remainderCol = (tensorInfo.chunkDimSize % tensorInfo.chunkCol) * tensorInfo.originCol;
                uint32_t blockLen = remainderCol > tensorInfo.startOffset ? remainderCol - tensorInfo.startOffset : 0;
                colLen0 = blockLen > srcLen ? srcLen : blockLen;
                colLen1 = srcLen - colLen0;
                colLoop1 = colLen1 / mainFP32;
                tailPad = colLen1 - colLoop1 * mainFP32;
            }
        }
        uint32_t padLen = isSplit ? this->GetAlign(srcLen * rowLoop0, this->srcEleUbBlock_) - srcLen * rowLoop0 : 0;
        uint32_t dstOffset = totalCol + tensorInfo.startOffset - ubLoopInfo.colStart - ubLoopInfo.preCatCol;

        auto dstAddr = (__ubuf__ T2*)this->dstLocal_.GetPhyAddr() + dstOffset;
        auto srcAddr = (__ubuf__ T1*)this->srcLocal_.GetPhyAddr() + localOffset;

        __VEC_SCOPE__
        {
            // rowLoop0
            DoCopyCatVF(dstAddr, srcAddr, rowLoop0, colLoop, tail, rowStride);
            // rowLoop2
            DoPadCatVF(dstAddr + (rowLoop0 + rowLoop1) * rowStride, rowLoop2, colLoopFP32, mainFP32, tailFP32, rowStride);
            // rowLoop1
            DoCopyCatVF(dstAddr + rowLoop0 * rowStride, srcAddr + rowLoop0 * srcLen + padLen, rowLoop1, colLoop, tail, rowStride);
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_STORE>();
            DoPadCatVF(dstAddr + rowLoop0 * rowStride + colLen0, rowLoop1, colLoop1, mainFP32, tailPad, rowStride);
        }
    }
};
#endif // _CHUNK_CAT_ARCH35_H_