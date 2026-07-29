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
 * \file radix_sort_topk_b16.h
 * \brief radix_sort_topk_b16 impl
 */
#ifndef RADIX_SORT_TOPK_B16_H
#define RADIX_SORT_TOPK_B16_H
#include "kernel_operator.h"
#include "top_k_util_type_simd.h"
using namespace AscendC;
using namespace topkV2;
template <typename T, typename UNSIGNED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
struct RadixSortTopKB16{
    __aicore__ inline RadixSortTopKB16() {}
    __aicore__ inline void GetCumSum(
        LocalTensor<UNSIGNED_TYPE> inputX,
        LocalTensor<int32_t> blockCumSum,
        UNSIGNED_TYPE andDataMask,
        UNSIGNED_TYPE involvedDataMask,
        uint16_t round,
        uint32_t numTileData);
};

template <typename T, typename UNSIGNED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
__aicore__ inline void RadixSortTopKB16<T, UNSIGNED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX>::GetCumSum(
    LocalTensor<UNSIGNED_TYPE> inputX,
    LocalTensor<int32_t> blockCumSum,
    UNSIGNED_TYPE andDataMask,
    UNSIGNED_TYPE involvedDataMask,
    uint16_t round,
    uint32_t numTileData)
{
    __local_mem__ UNSIGNED_TYPE* inputXValuePtr = (__ubuf__ UNSIGNED_TYPE*)inputX.GetPhyAddr();
    __local_mem__ int32_t* blockCumSumPtr = (__ubuf__ int32_t*)blockCumSum.GetPhyAddr();
    int16_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repateTime = (numTileData + ONE_TIMES_B16_NUM - 1) / ONE_TIMES_B16_NUM;
    uint16_t roundValue = NUM_PASS - 1;
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint16_t> inputVectorOne;
        MicroAPI::RegTensor<uint16_t> chistVectorZero, chistVectorOne;
        MicroAPI::RegTensor<uint16_t> zeroVector;
        MicroAPI::RegTensor<uint16_t> andMaskVector;
        MicroAPI::RegTensor<uint16_t> firstRoundVector;
        MicroAPI::MaskReg predicateDefaultB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg predicateDefaultB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(zeroVector, 0, predicateDefaultB16);
        MicroAPI::Duplicate(chistVectorZero, 0, predicateDefaultB16);
        MicroAPI::Duplicate(chistVectorOne, 0, predicateDefaultB16);
        MicroAPI::Duplicate(andMaskVector, andDataMask, predicateDefaultB16);
        MicroAPI::Duplicate(firstRoundVector, roundValue, predicateDefaultB16);
        uint32_t inputElementNum = numTileData;
        for (uint16_t i = 0; i < repateTime; i++) {
            MicroAPI::MaskReg histMask = MicroAPI::UpdateMask<uint16_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr, ONE_TIMES_B16_NUM);
            // get high bit value
            MicroAPI::RegTensor<uint16_t> andValueZero;
            MicroAPI::And(andValueZero, inputVectorOne, andMaskVector, histMask);
            // get same value
            MicroAPI::MaskReg cmpValueMask;
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(cmpValueMask, andValueZero, involvedDataMask, histMask);
            // update mask for next step calc
            MicroAPI::MaskReg newHistMask;
            MicroAPI::MaskAnd(newHistMask, cmpValueMask, histMask, predicateDefaultB16);
            // is first round
            MicroAPI::MaskReg finalMask, firstRoundMask;
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(firstRoundMask, firstRoundVector, round, histMask);
            // mask sel
            MicroAPI::MaskSel(finalMask, histMask, newHistMask, firstRoundMask);
            // vshr
            MicroAPI::RegTensor<uint16_t> shiftVecOne;
            MicroAPI::ShiftRights<uint16_t, int16_t>(shiftVecOne, inputVectorOne, bitOffset, finalMask);
            // get 256 8bit
            MicroAPI::RegTensor<uint8_t> shiftVecU8LowBit, shiftVecU8HighBit;
            MicroAPI::DeInterleave(shiftVecU8LowBit, shiftVecU8HighBit, (MicroAPI::RegTensor<uint8_t> &)shiftVecOne, (MicroAPI::RegTensor<uint8_t> &)zeroVector);
            // copy u16 mask
            MicroAPI::MaskReg maskU8;
            MicroAPI::MaskPack(maskU8, finalMask);
            // get hist
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::ACCUMULATE>(chistVectorZero, shiftVecU8LowBit, maskU8);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::ACCUMULATE>(chistVectorOne, shiftVecU8LowBit, maskU8);
        }
        StoreCumsumToUB<int32_t, uint16_t>(blockCumSumPtr, chistVectorZero, chistVectorOne, zeroVector);
    }
}
#endif