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
 * \file radix_sort_topk_b8.h
 * \brief radix_sort_topk_b8 impl
 */
#ifndef RADIX_SORT_TOPK_B8_H
#define RADIX_SORT_TOPK_B8_H
#include "kernel_operator.h"
#include "top_k_util_type_simd.h"
using namespace AscendC;
using namespace topkV2;
template <typename T, typename UNSIGNED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
struct RadixSortTopKB8{
    __aicore__ inline RadixSortTopKB8() {}
    __aicore__ inline void GetCumSum(
        LocalTensor<UNSIGNED_TYPE> inputX,
        LocalTensor<int32_t> blockCumSum,
        UNSIGNED_TYPE andDataMask,
        UNSIGNED_TYPE involvedDataMask,
        int32_t round,
        uint32_t numTileData);
};

template <typename T, typename U>
__aicore__ inline void StoreCumsumToUB(
    __local_mem__ T* blockCumSumPtr, MicroAPI::RegTensor<U>& chistVectorZero,
    MicroAPI::RegTensor<U>& chistVectorOne, MicroAPI::RegTensor<U>& zeroVector)
{
    __VEC_SCOPE__{
        MicroAPI::MaskReg predicateDefaultB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::RegTensor<int32_t> cumSumZeroB32, cumSumOneB32;
        MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)cumSumZeroB32, (MicroAPI::RegTensor<uint16_t> &)cumSumOneB32, chistVectorZero, zeroVector);
        MicroAPI::RegTensor<int32_t> cumSumTwoB32, cumSumThreeB32;
        MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)cumSumTwoB32, (MicroAPI::RegTensor<uint16_t> &)cumSumThreeB32, chistVectorOne, zeroVector);
        MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockCumSumPtr, cumSumZeroB32, ONE_TIMES_B32_NUM, predicateDefaultB32);
        MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockCumSumPtr, cumSumOneB32, ONE_TIMES_B32_NUM, predicateDefaultB32);
        MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockCumSumPtr, cumSumTwoB32, ONE_TIMES_B32_NUM, predicateDefaultB32);
        MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockCumSumPtr, cumSumThreeB32, ONE_TIMES_B32_NUM, predicateDefaultB32);
    }
}

template <typename T, typename UNSIGNED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
__aicore__ inline void RadixSortTopKB8<T, UNSIGNED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX>::GetCumSum(
    LocalTensor<UNSIGNED_TYPE> inputX,
    LocalTensor<int32_t> blockCumSum,
    UNSIGNED_TYPE andDataMask,
    UNSIGNED_TYPE involvedDataMask,
    int32_t round,
    uint32_t numTileData)
{
    __local_mem__ UNSIGNED_TYPE* inputXValuePtr = (__ubuf__ UNSIGNED_TYPE*)inputX.GetPhyAddr();
    __local_mem__ int32_t* blockCumSumPtr = (__ubuf__ int32_t*)blockCumSum.GetPhyAddr();
    int16_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repateTime = (numTileData + ONE_TIMES_B8_NUM - 1) / ONE_TIMES_B8_NUM;
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint8_t> inputVectorOne;
        MicroAPI::RegTensor<uint16_t> chistVectorZero, chistVectorOne;
        MicroAPI::RegTensor<uint16_t> zeroVector;
        MicroAPI::MaskReg predicateDefaultB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::MaskReg predicateDefaultB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::Duplicate(chistVectorZero, 0, predicateDefaultB16);
        MicroAPI::Duplicate(chistVectorOne, 0, predicateDefaultB16);
        MicroAPI::Duplicate(zeroVector, 0, predicateDefaultB16);
        uint32_t inputElementNum = numTileData;
        for (uint16_t i = 0; i < repateTime; i++) {
            MicroAPI::MaskReg histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr, ONE_TIMES_B8_NUM);
            // get hist
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::ACCUMULATE>(chistVectorZero, inputVectorOne, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::ACCUMULATE>(chistVectorOne, inputVectorOne, histMask);
        }
        StoreCumsumToUB<int32_t, uint16_t>(blockCumSumPtr, chistVectorZero, chistVectorOne, zeroVector);
    }
}
#endif