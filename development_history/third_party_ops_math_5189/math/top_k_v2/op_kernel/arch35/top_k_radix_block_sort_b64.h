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
 * \file top_k_radix_block_sort_b64.h
 * \brief topk sort b64 impl
 */
#ifndef TOP_K_RADIX_BLOCK_SORT_B64_H
#define TOP_K_RADIX_BLOCK_SORT_B64_H
#include "kernel_operator.h"
#include "top_k_util_type_simd.h"
#include "top_k_constant_var_simd.h"
using namespace AscendC;
using namespace topkV2;
template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
struct RadixBlockSortSimdB64 {
public:
    __aicore__ inline RadixBlockSortSimdB64() {}
    __aicore__ inline void GetGlobalExcusiveSum(
        LocalTensor<uint64_t> inputX,
        LocalTensor<T_INDEX> blockExcusive,
        uint32_t numTileData);
    __aicore__ inline void GetBlockExcusiveSum(
        LocalTensor<uint64_t> inputX,
        LocalTensor<uint8_t> inputXBitValue,
        LocalTensor<uint8_t> inputXBitValueCopy,
        LocalTensor<uint16_t> blockExcusive,
        LocalTensor<uint16_t> blockHist,
        int32_t round,
        uint32_t numTileData);
    __aicore__ inline void TwiddleInB64(
        LocalTensor<T> inputX,
        LocalTensor<UNSINGED_TYPE> uintInputX,
        uint32_t numTileData);
    __aicore__ inline void ReverseInputData(
        LocalTensor<UNSINGED_TYPE> inputX,
        LocalTensor<UNSINGED_TYPE> reverseInputX,
        uint32_t numTileData);
};

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
__aicore__ inline void RadixBlockSortSimdB64<T, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX>::GetGlobalExcusiveSum(
    LocalTensor<uint64_t> inputX,
    LocalTensor<T_INDEX> blockExcusive,
    uint32_t numTileData)
{
    __local_mem__ uint64_t* inputXValuePtr = (__ubuf__ uint64_t*)inputX.GetPhyAddr();
    __local_mem__ T_INDEX* blockExcusivePtr = (__ubuf__ T_INDEX*)blockExcusive.GetPhyAddr();
    __local_mem__ T_INDEX* blockExcusivePtrRead = blockExcusivePtr;
    __local_mem__ T_INDEX* blockExcusivePtrWrite = blockExcusivePtr;
    uint16_t loopTime = NUM_PASS;
    uint16_t repeatTime = (numTileData + ONE_TIMES_B64_NUM - 1) / ONE_TIMES_B64_NUM;
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint64_t> inputVectorOne;
        MicroAPI::RegTensor<uint16_t> histVectorZero, histVectorOne;
        MicroAPI::RegTensor<uint16_t> chistVectorZero, chistVectorOne;
        MicroAPI::RegTensor<uint64_t> zeroVectorU64;
        MicroAPI::RegTensor<uint32_t> zeroVectorU32;
        MicroAPI::RegTensor<uint16_t> zeroVector;
        MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint64_t>();
        MicroAPI::MaskReg predicateDefaultB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg predicateDefaultB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(zeroVectorU64, 0, predicateDefault);
        MicroAPI::Duplicate(zeroVectorU32, 0, predicateDefaultB32);
        MicroAPI::Duplicate(zeroVector, 0, predicateDefaultB16);
        for(uint16_t round = 0; round < loopTime; round++) {
            MicroAPI::Duplicate(histVectorZero, 0, predicateDefaultB16);
            MicroAPI::Duplicate(histVectorOne, 0, predicateDefaultB16);
            MicroAPI::Duplicate(chistVectorZero, 0, predicateDefaultB16);
            MicroAPI::Duplicate(chistVectorOne, 0, predicateDefaultB16);
            uint32_t inputElementNum = numTileData;
            int16_t bitOffset = round * SHIFT_BIT_NUM;
            __local_mem__ uint64_t* inputXValuePtrCopy = inputXValuePtr;
            // calc hist/excusive
            for (uint16_t i = 0; i < repeatTime; i++) {
                MicroAPI::MaskReg histMask = MicroAPI::UpdateMask<uint64_t>(inputElementNum);
                // load input
                MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtrCopy, ONE_TIMES_B64_NUM);
                // vshr
                MicroAPI::RegTensor<uint64_t> shiftVecOne;
                MicroAPI::ShiftRights<uint64_t, int16_t>(shiftVecOne, inputVectorOne, bitOffset, predicateDefault);
                // get u64 low bit
                MicroAPI::RegTensor<uint32_t> shiftVecU32LowBitOne, shiftVecU32HighBitOne;
                MicroAPI::DeInterleave(shiftVecU32LowBitOne, shiftVecU32HighBitOne, (MicroAPI::RegTensor<uint32_t> &)shiftVecOne, (MicroAPI::RegTensor<uint32_t> &)zeroVectorU64);
                // get u32 low bit
                MicroAPI::RegTensor<uint16_t> shiftVecU16LowBitOne, shiftVecU16HighBitOne;
                MicroAPI::DeInterleave(shiftVecU16LowBitOne, shiftVecU16HighBitOne, (MicroAPI::RegTensor<uint16_t> &)shiftVecU32LowBitOne, (MicroAPI::RegTensor<uint16_t> &)zeroVectorU32);
                // get u16 low bit
                MicroAPI::RegTensor<uint8_t> shiftVecU8LowBit, shiftVecU8HighBit;
                MicroAPI::DeInterleave(shiftVecU8LowBit, shiftVecU8HighBit, (MicroAPI::RegTensor<uint8_t> &)shiftVecU16LowBitOne, (MicroAPI::RegTensor<uint8_t> &)zeroVector);
                // copy u32 mask
                MicroAPI::MaskReg maskU8, maskU16, maskU32;
                MicroAPI::MaskPack(maskU32, histMask);
                MicroAPI::MaskPack(maskU16, maskU32);
                MicroAPI::MaskPack(maskU8, maskU16);
                // get hist
                MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                    MicroAPI::HistogramsType::FREQUENCY>(histVectorZero, shiftVecU8LowBit, maskU8);
                MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                    MicroAPI::HistogramsType::FREQUENCY>(histVectorOne, shiftVecU8LowBit, maskU8);
                // get cusum
                MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                    MicroAPI::HistogramsType::ACCUMULATE>(chistVectorZero, shiftVecU8LowBit, maskU8);
                MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                    MicroAPI::HistogramsType::ACCUMULATE>(chistVectorOne, shiftVecU8LowBit, maskU8);
            }
            // get excusive sum
            MicroAPI::RegTensor<uint16_t> excusiveSumZero, excusiveSumOne;
            MicroAPI::Sub(excusiveSumZero, chistVectorZero, histVectorZero, predicateDefaultB16);
            MicroAPI::Sub(excusiveSumOne, chistVectorOne, histVectorOne, predicateDefaultB16);
            // case B16 to B32
            MicroAPI::RegTensor<int32_t> excusiveSumZeroB32, excusiveSumOneB32, excusiveSumTwoB32, excusiveSumThreeB32;
            MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)excusiveSumZeroB32, (MicroAPI::RegTensor<uint16_t> &)excusiveSumOneB32, excusiveSumZero, zeroVector);
            MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)excusiveSumTwoB32, (MicroAPI::RegTensor<uint16_t> &)excusiveSumThreeB32, excusiveSumOne, zeroVector);
            if constexpr (IsSameType<T_INDEX, int32_t>::value) {
                // load global excusive
                MicroAPI::RegTensor<int32_t> excusiveSumGlobalZero, excusiveSumGlobalOne, excusiveSumGlobalTwo, excusiveSumGlobalThree;
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalZero, blockExcusivePtrRead, ONE_TIMES_B32_NUM);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalOne, blockExcusivePtrRead, ONE_TIMES_B32_NUM);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalTwo, blockExcusivePtrRead, ONE_TIMES_B32_NUM);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalThree, blockExcusivePtrRead, ONE_TIMES_B32_NUM);
                // add block ans to global excusive
                MicroAPI::Add(excusiveSumGlobalZero, excusiveSumGlobalZero, excusiveSumZeroB32, predicateDefaultB32);
                MicroAPI::Add(excusiveSumGlobalOne, excusiveSumGlobalOne, excusiveSumOneB32, predicateDefaultB32);
                MicroAPI::Add(excusiveSumGlobalTwo, excusiveSumGlobalTwo, excusiveSumTwoB32, predicateDefaultB32);
                MicroAPI::Add(excusiveSumGlobalThree, excusiveSumGlobalThree, excusiveSumThreeB32, predicateDefaultB32);
                // vsts to global
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalZero, ONE_TIMES_B32_NUM, predicateDefaultB32);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalOne, ONE_TIMES_B32_NUM, predicateDefaultB32);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalTwo, ONE_TIMES_B32_NUM, predicateDefaultB32);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalThree, ONE_TIMES_B32_NUM, predicateDefaultB32);
            } else {
                // cast B32 to B64
                MicroAPI::RegTensor<int64_t> excusiveSumZeroB64A, excusiveSumZeroB64B, excusiveSumOneB64A, excusiveSumOneB64B;
                MicroAPI::RegTensor<int64_t> excusiveSumTwoB64A, excusiveSumTwoB64B, excusiveSumThreeB64A, excusiveSumThreeB64B;
                MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)excusiveSumZeroB64A, (MicroAPI::RegTensor<uint16_t> &)excusiveSumZeroB64B,
                                    (MicroAPI::RegTensor<uint16_t> &)excusiveSumZeroB32, zeroVector);
                MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)excusiveSumOneB64A, (MicroAPI::RegTensor<uint16_t> &)excusiveSumOneB64B,
                                    (MicroAPI::RegTensor<uint16_t> &)excusiveSumOneB32, zeroVector);
                MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)excusiveSumTwoB64A, (MicroAPI::RegTensor<uint16_t> &)excusiveSumTwoB64B,
                                    (MicroAPI::RegTensor<uint16_t> &)excusiveSumTwoB32, zeroVector);
                MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)excusiveSumThreeB64A, (MicroAPI::RegTensor<uint16_t> &)excusiveSumThreeB64B,
                                    (MicroAPI::RegTensor<uint16_t> &)excusiveSumThreeB32, zeroVector);
                // load global excusive
                MicroAPI::RegTensor<int64_t> excusiveSumGlobalZeroA, excusiveSumGlobalZeroB, excusiveSumGlobalOneA, excusiveSumGlobalOneB;
                MicroAPI::RegTensor<int64_t> excusiveSumGlobalTwoA, excusiveSumGlobalTwoB, excusiveSumGlobalThreeA, excusiveSumGlobalThreeB;
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalZeroA, blockExcusivePtrRead, ONE_TIMES_B64_NUM);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalZeroB, blockExcusivePtrRead, ONE_TIMES_B64_NUM);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalOneA, blockExcusivePtrRead, ONE_TIMES_B64_NUM);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalOneB, blockExcusivePtrRead, ONE_TIMES_B64_NUM);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalTwoA, blockExcusivePtrRead, ONE_TIMES_B64_NUM);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalTwoB, blockExcusivePtrRead, ONE_TIMES_B64_NUM);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalThreeA, blockExcusivePtrRead, ONE_TIMES_B64_NUM);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(excusiveSumGlobalThreeB, blockExcusivePtrRead, ONE_TIMES_B64_NUM);
                // add block ans to global excusive
                MicroAPI::Add(excusiveSumGlobalZeroA, excusiveSumGlobalZeroA, excusiveSumZeroB64A, predicateDefault);
                MicroAPI::Add(excusiveSumGlobalZeroB, excusiveSumGlobalZeroB, excusiveSumZeroB64B, predicateDefault);
                MicroAPI::Add(excusiveSumGlobalOneA, excusiveSumGlobalOneA, excusiveSumOneB64A, predicateDefault);
                MicroAPI::Add(excusiveSumGlobalOneB, excusiveSumGlobalOneB, excusiveSumOneB64B, predicateDefault);
                MicroAPI::Add(excusiveSumGlobalTwoA, excusiveSumGlobalTwoA, excusiveSumTwoB64A, predicateDefault);
                MicroAPI::Add(excusiveSumGlobalTwoB, excusiveSumGlobalTwoB, excusiveSumTwoB64B, predicateDefault);
                MicroAPI::Add(excusiveSumGlobalThreeA, excusiveSumGlobalThreeA, excusiveSumThreeB64A, predicateDefault);
                MicroAPI::Add(excusiveSumGlobalThreeB, excusiveSumGlobalThreeB, excusiveSumThreeB64B, predicateDefault);
                // vsts to global
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalZeroA, ONE_TIMES_B64_NUM, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalZeroB, ONE_TIMES_B64_NUM, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalOneA, ONE_TIMES_B64_NUM, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalOneB, ONE_TIMES_B64_NUM, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalTwoA, ONE_TIMES_B64_NUM, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalTwoB, ONE_TIMES_B64_NUM, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalThreeA, ONE_TIMES_B64_NUM, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusivePtrWrite, excusiveSumGlobalThreeB, ONE_TIMES_B64_NUM, predicateDefault);
            }
        }
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
__aicore__ inline void RadixBlockSortSimdB64<T, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX>::GetBlockExcusiveSum(
    LocalTensor<uint64_t> inputX,
    LocalTensor<uint8_t> inputXBitValue,
    LocalTensor<uint8_t> inputXBitValueCopy,
    LocalTensor<uint16_t> blockExcusive,
    LocalTensor<uint16_t> blockHist,
    int32_t round,
    uint32_t numTileData)
{
    int16_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = (numTileData + ONE_TIMES_B64_NUM - 1) / ONE_TIMES_B64_NUM;
    __local_mem__ uint64_t* inputXValuePtr = (__ubuf__ uint64_t*)inputX.GetPhyAddr();
    __local_mem__ uint8_t* inputX8BitValuePtr = (__ubuf__ uint8_t*)inputXBitValue.GetPhyAddr();
    __local_mem__ uint8_t* inputX8BitValueCopyPtr = (__ubuf__ uint8_t*)inputXBitValueCopy.GetPhyAddr();
    __local_mem__ uint16_t* blockExcusiveLocalPtr = (__ubuf__ uint16_t*)blockExcusive.GetPhyAddr();
    __local_mem__ uint16_t* blockHistPtr = (__ubuf__ uint16_t*)blockHist.GetPhyAddr();
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint64_t> inputVectorOne;
        MicroAPI::RegTensor<uint16_t> histVectorZero, histVectorOne;
        MicroAPI::RegTensor<uint16_t> chistVectorZero, chistVectorOne;
        MicroAPI::RegTensor<uint64_t> zeroVectorU64;
        MicroAPI::RegTensor<uint32_t> zeroVectorU32;
        MicroAPI::RegTensor<uint16_t> zeroVector;
        MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint64_t>();
        MicroAPI::MaskReg predicateDefaultB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg predicateDefaultB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(zeroVectorU64, 0, predicateDefault);
        MicroAPI::Duplicate(zeroVectorU32, 0, predicateDefaultB32);
        MicroAPI::Duplicate(zeroVector, 0, predicateDefaultB16);
        MicroAPI::Duplicate(histVectorZero, 0, predicateDefaultB16);
        MicroAPI::Duplicate(histVectorOne, 0, predicateDefaultB16);
        MicroAPI::Duplicate(chistVectorZero, 0, predicateDefaultB16);
        MicroAPI::Duplicate(chistVectorOne, 0, predicateDefaultB16);
        uint32_t inputElementNum = numTileData;
        for (uint16_t i = 0; i < repeatTime; i++) {
            MicroAPI::MaskReg histMask = MicroAPI::UpdateMask<uint64_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr, ONE_TIMES_B64_NUM);
            // vshr
            MicroAPI::RegTensor<uint64_t> shiftVecOne;
            MicroAPI::ShiftRights<uint64_t, int16_t>(shiftVecOne, inputVectorOne, bitOffset, predicateDefault);
            // get u64 low bit
            MicroAPI::RegTensor<uint32_t> shiftVecU32LowBitOne, shiftVecU32HighBitOne;
            MicroAPI::DeInterleave(shiftVecU32LowBitOne, shiftVecU32HighBitOne, (MicroAPI::RegTensor<uint32_t> &)shiftVecOne, (MicroAPI::RegTensor<uint32_t> &)zeroVectorU64);
            // get u32 low bit
            MicroAPI::RegTensor<uint16_t> shiftVecU16LowBitOne, shiftVecU16HighBitOne;
            MicroAPI::DeInterleave(shiftVecU16LowBitOne, shiftVecU16HighBitOne, (MicroAPI::RegTensor<uint16_t> &)shiftVecU32LowBitOne, (MicroAPI::RegTensor<uint16_t> &)zeroVectorU32);
            // get u16 low bit
            MicroAPI::RegTensor<uint8_t> shiftVecU8LowBit, shiftVecU8HighBit;
            MicroAPI::DeInterleave(shiftVecU8LowBit, shiftVecU8HighBit, (MicroAPI::RegTensor<uint8_t> &)shiftVecU16LowBitOne, (MicroAPI::RegTensor<uint8_t> &)zeroVector);
            // copy u32 mask
            MicroAPI::MaskReg maskU8, maskU16, maskU32;
            MicroAPI::MaskPack(maskU32, histMask);
            MicroAPI::MaskPack(maskU16, maskU32);
            MicroAPI::MaskPack(maskU8, maskU16);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputX8BitValuePtr, shiftVecU8LowBit, ONE_TIMES_B64_NUM, maskU8);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputX8BitValueCopyPtr, shiftVecU8LowBit, ONE_TIMES_B64_NUM, maskU8);
            // get hist
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                    MicroAPI::HistogramsType::FREQUENCY>(histVectorZero, shiftVecU8LowBit, maskU8);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                    MicroAPI::HistogramsType::FREQUENCY>(histVectorOne, shiftVecU8LowBit, maskU8);
            // get cusum
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                    MicroAPI::HistogramsType::ACCUMULATE>(chistVectorZero, shiftVecU8LowBit, maskU8);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                    MicroAPI::HistogramsType::ACCUMULATE>(chistVectorOne, shiftVecU8LowBit, maskU8);
        }
        // get excusive sum
        MicroAPI::RegTensor<uint16_t> excusiveSumZero, excusiveSumOne;
        MicroAPI::Sub(excusiveSumZero, chistVectorZero, histVectorZero, predicateDefaultB16);
        MicroAPI::Sub(excusiveSumOne, chistVectorOne, histVectorOne, predicateDefaultB16);
        // store excusive sum to ub
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveLocalPtr, excusiveSumZero, ONE_TIMES_B16_NUM, predicateDefaultB16);
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveLocalPtr, excusiveSumOne, ONE_TIMES_B16_NUM, predicateDefaultB16);
        // store hist to ub
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistPtr, histVectorZero, ONE_TIMES_B16_NUM, predicateDefaultB16);
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistPtr, histVectorOne, ONE_TIMES_B16_NUM, predicateDefaultB16);
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
__aicore__ inline void RadixBlockSortSimdB64<T, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX>::TwiddleInB64(
    LocalTensor<T> inputX,
    LocalTensor<UNSINGED_TYPE> uintInputX,
    uint32_t numTileData)
{
    __local_mem__ UNSINGED_TYPE* inputXValuePtr = (__ubuf__ UNSINGED_TYPE*)inputX.GetPhyAddr();
    __local_mem__ UNSINGED_TYPE* uinputXValuePtr = (__ubuf__ UNSINGED_TYPE*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = (numTileData + ONE_TIMES_B64_NUM - 1) / ONE_TIMES_B64_NUM;
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint64_t> inputVectorZero;
        MicroAPI::RegTensor<uint64_t> xorValueVectorZero;
        MicroAPI::MaskReg predicateDefaultB64 = MicroAPI::CreateMask<uint64_t>();
        MicroAPI::Duplicate(xorValueVectorZero, XOR_OP_VALUE_B64, predicateDefaultB64);
        uint32_t inputElementNum = numTileData;
        for (uint16_t i = 0; i < repeatTime; i++) {
            MicroAPI::MaskReg xorMask = MicroAPI::UpdateMask<uint64_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputVectorZero, inputXValuePtr, ONE_TIMES_B64_NUM);
            // vxor
            MicroAPI::RegTensor<uint64_t> vstVectorZero;
            MicroAPI::Xor(vstVectorZero, inputVectorZero, xorValueVectorZero, predicateDefaultB64);
            // sts
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uinputXValuePtr, vstVectorZero, ONE_TIMES_B64_NUM, xorMask);
        }
    }
    if (IS_DESCEND) {
        ReverseInputData(uintInputX, uintInputX, numTileData);
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
__aicore__ inline void RadixBlockSortSimdB64<T, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX>::ReverseInputData(
    LocalTensor<UNSINGED_TYPE> inputX,
    LocalTensor<UNSINGED_TYPE> reverseInputX,
    uint32_t numTileData){
    __local_mem__ UNSINGED_TYPE* inputXValuePtr = (__ubuf__ UNSINGED_TYPE*)inputX.GetPhyAddr();
    __local_mem__ UNSINGED_TYPE* inputXValuePtrCopy = inputXValuePtr;
    __local_mem__ UNSINGED_TYPE* reverseInputXPtr = (__ubuf__ UNSINGED_TYPE*)reverseInputX.GetPhyAddr();
    uint16_t repeatTime = (numTileData + ONE_TIMES_B64_NUM - 1) / ONE_TIMES_B64_NUM;
    __VEC_SCOPE__ {
        uint32_t inputElementNum = numTileData;
        MicroAPI::RegTensor<uint64_t> inputVectorOne;
        MicroAPI::RegTensor<uint64_t> vnotVectorZero;
        MicroAPI::MaskReg predicateDefaultB64 = MicroAPI::CreateMask<uint64_t>();
        for (uint16_t i = 0; i < repeatTime; i++) {
            MicroAPI::MaskReg vnotMask = MicroAPI::UpdateMask<uint64_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtrCopy, ONE_TIMES_B64_NUM);
            // ~
            MicroAPI::Not(vnotVectorZero, inputVectorOne, predicateDefaultB64);
            // sts
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reverseInputXPtr, vnotVectorZero, ONE_TIMES_B64_NUM, vnotMask);
        }
    }
}
#endif