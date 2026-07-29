/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arg_max_with_value_base.h
 * \brief arg_max_with_value_base
 */

#ifndef ARG_MAX_WITH_VALUE_BASE_H
#define ARG_MAX_WITH_VALUE_BASE_H

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace ArgMaxWithValue {
using namespace AscendC;
constexpr int64_t NDDMA_DIM_NUM = 3;
constexpr uint16_t VALUE_TWO = 2;
constexpr uint16_t VALUE_ONE = 1;
constexpr uint16_t VALUE_ZERO = 0;
constexpr uint32_t BLOCK_SIZE = 32;

constexpr float MAX_VALUE_FLOAT = INFINITY;
constexpr int32_t MAX_VALUE_INT32 = INT32_MAX;
constexpr int64_t MAX_VALUE_INT64 = INT64_MAX;
constexpr float MIN_VALUE_FLOAT = -INFINITY;
constexpr int32_t MIN_VALUE_INT32 = INT32_MIN;
constexpr int64_t MIN_VALUE_INT64 = INT64_MIN;

template <typename T1, typename T2, typename T3, bool isMin> class ArgMaxWithValueBase {
public:
    __aicore__ inline ArgMaxWithValueBase(){};

protected:
    template <typename T4> __aicore__ inline T4 GetMaxOrMinValue()
    {
        T4 minValue = 0;
        // isMin时取最大值，isMax时取最小值
        if constexpr (IsSameType<T4, float>::value) {
            minValue = isMin ? MAX_VALUE_FLOAT : MIN_VALUE_FLOAT;
        } else if constexpr (IsSameType<T4, half>::value) {
            half MAX_VALUE_HALF = INFINITY;
            half MIN_VALUE_HALF = -INFINITY;
            minValue = isMin ? MAX_VALUE_HALF : MIN_VALUE_HALF;
        } else if constexpr (IsSameType<T4, int32_t>::value) {
            minValue = isMin ? MAX_VALUE_INT32 : MIN_VALUE_INT32;
        } else if constexpr (IsSameType<T4, int64_t>::value) {
            minValue = isMin ? MAX_VALUE_INT64 : MIN_VALUE_INT64;
        }
        return minValue;
    };

    __aicore__ inline uint16_t GetVLSize()
    {
        uint16_t vlSize = 1;
        if constexpr (IsSameType<T1, int64_t>::value) {
            vlSize = Ops::Base::GetVRegSize() / sizeof(T1) * VALUE_TWO;
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            vlSize = Ops::Base::GetVRegSize() / sizeof(float);
        } else {
            vlSize = Ops::Base::GetVRegSize() / sizeof(T1);
        }
        return vlSize;
    };
    template <typename T4, typename T5, bool needAlign = false>
    __aicore__ inline void ArgMaxV1(__local_mem__ void *valueDst, __local_mem__ void *indexDst, __local_mem__ void *src,
        uint16_t dimA, uint16_t dimR, uint32_t startR = 0);
    template <typename T4, typename T5, typename T6, bool needAlign = false>
    __aicore__ inline void ArgMaxGatherV1(__local_mem__ void *valueDst, __local_mem__ void *indexDst,
        __local_mem__ void *src, uint16_t dimA, uint16_t dimR, uint16_t dimNextA, uint32_t startR = 0);
    template <typename T4, typename T5, typename T6, bool needAlign = false>
    __aicore__ inline void ArgMaxGatherV1ForRLessThanVL64(__local_mem__ void *valueDst, __local_mem__ void *indexDst,
        __local_mem__ void *src, uint16_t dimA, uint16_t dimR, uint16_t dimNextA, uint32_t startR = 0);
    template <typename T4, typename T5, typename T6>
    __aicore__ inline void ArgMaxGatherRa(__local_mem__ void *valueDst, __local_mem__ T2 *indexDst,
        __local_mem__ void *src, uint16_t dimA, uint16_t dimR);
    template <typename T4, typename T5, bool isBfloat16Casted = true>
    __aicore__ inline void ArgMaxRa(__local_mem__ void *valueDst, __local_mem__ T2 *indexDst, __local_mem__ void *src,
        uint16_t dimA, uint16_t alignA, uint16_t dimR, uint32_t startR = 0);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxAra(
        __local_mem__ T4* valueDst, __local_mem__ T2* indexDst, __local_mem__ T4* src, uint16_t dimA0,
        uint16_t dimR, uint16_t dimA1, uint32_t startR);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxAraInt64(__local_mem__ T4 *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ T4 *src, uint16_t dimA0, uint16_t dimR, uint16_t dimA1,
    uint32_t startR);
    template <typename T4, bool isBfloat16Casted = true>
    __aicore__ inline void UpdateResult(__local_mem__ void *srcIndice, __local_mem__ void *srcValue,
        __local_mem__ void *dstIndice, __local_mem__ void *dstValue, uint64_t num);
#if ORIG_DTYPE_X == DT_INT64
    template <bool needAlign = false>
    __aicore__ inline void ArgMaxV1Int64(__local_mem__ T1 *valueDst, __local_mem__ T2 *indexDst, __local_mem__ T1 *src,
        uint16_t dimA, uint16_t dimR, uint32_t startR = 0);
    template <bool needAlign = false>
    __aicore__ inline void ArgMaxV1Int64ForRLessThanVL64(__local_mem__ T1 *valueDst, __local_mem__ T2 *indexDst, __local_mem__ T1 *src,
        uint16_t dimA, uint16_t dimR, uint32_t startR = 0);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxRaInt64(__local_mem__ void *valueDst, __local_mem__ T2 *indexDst, __local_mem__ void *src,
        uint16_t dimA, uint16_t alignA, uint16_t dimR, uint32_t startR = 0);
    template <typename T4, typename T5, typename T6>
    __aicore__ inline void ArgMaxGatherRaInt64(__local_mem__ void *valueDst, __local_mem__ T2 *indexDst,
        __local_mem__ void *src, uint16_t dimA, uint16_t dimR);
    template <typename T4, typename T5, typename T6, bool needAlign = false>
    __aicore__ inline void ArgMaxGatherV1Int64(__local_mem__ void *valueDst, __local_mem__ void *indexDst,
        __local_mem__ void *src, uint16_t dimA, uint16_t dimR, uint16_t dimNextA, uint32_t startR = 0);
#endif
    __aicore__ inline void CastToBF16(LocalTensor<T1> &outValues, LocalTensor<float> outValues32, uint64_t copyNum);
    __aicore__ inline void CopyOutValues(GlobalTensor<T1> &valuesGm, LocalTensor<T1> &outValues, uint64_t offset,
        uint64_t copyNum);
    __aicore__ inline void CopyOutIndice(GlobalTensor<T3> &indiceGm, LocalTensor<T2> &outIndice, uint64_t offset,
        uint64_t copyNum, LocalTensor<T3> &castUb);
};

template <typename T1, typename T2, typename T3, bool isMin>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::CastToBF16(LocalTensor<T1> &outValues,
    LocalTensor<float> outValues32, uint64_t copyNum)
{
    Cast(outValues, outValues32, RoundMode::CAST_ROUND, copyNum);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
}

template <typename T1, typename T2, typename T3, bool isMin>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::CopyOutValues(GlobalTensor<T1> &valuesGm,
    LocalTensor<T1> &outValues, uint64_t offset, uint64_t copyNum)
{
    DataCopyExtParams copyOutValuesParams{ 1, 0, 0, 0, 0 };
    copyOutValuesParams.blockCount = 1;
    copyOutValuesParams.blockLen = copyNum * sizeof(T1);
    copyOutValuesParams.srcStride = 0;
    copyOutValuesParams.dstStride = 0;
    DataCopyPad(valuesGm[offset], outValues, copyOutValuesParams);
}

template <typename T1, typename T2, typename T3, bool isMin>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::CopyOutIndice(GlobalTensor<T3> &indiceGm,
    LocalTensor<T2> &outIndice, uint64_t offset, uint64_t copyNum, LocalTensor<T3> &castUb)
{
    DataCopyExtParams copyOutIndiceParams{ 1, 0, 0, 0, 0 };
    copyOutIndiceParams.blockCount = 1;
    copyOutIndiceParams.blockLen = copyNum * sizeof(T3);
    copyOutIndiceParams.srcStride = 0;
    copyOutIndiceParams.dstStride = 0;
    if constexpr (IsSameType<T1, int64_t>::value && !IsSameType<T3, int64_t>::value) {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Cast(castUb, outIndice, RoundMode::CAST_NONE, copyNum);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(indiceGm[offset], castUb, copyOutIndiceParams);
    } else if constexpr (!IsSameType<T1, int64_t>::value && IsSameType<T3, int64_t>::value) {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Cast(castUb, outIndice, RoundMode::CAST_NONE, copyNum);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(indiceGm[offset], castUb, copyOutIndiceParams);
    } else {
        DataCopyPad(indiceGm[offset], outIndice, copyOutIndiceParams);
    }
}

#if ORIG_DTYPE_X == DT_INT64
template <typename T1, typename T2, typename T3, bool isMin>
template <bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxV1Int64(__local_mem__ T1 *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ T1 *src, uint16_t dimA, uint16_t dimR, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    if (dimR < VL) {
        ArgMaxV1Int64ForRLessThanVL64<needAlign>(valueDst, indexDst, src, dimA, dimR, startR);
        return;
    }
    uint16_t mainLoop = dimR / VL;
    uint32_t tailSize = dimR - mainLoop * VL;
    uint16_t loopTime = mainLoop - 1;
    uint16_t tailLoop = tailSize > 0 ? 1 : 0;

    uint32_t offset1 = VL;
    uint32_t offsetTail = tailSize;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        offset1 = CeilDivision(offset1, alignNum) * alignNum;
        offsetTail = CeilDivision(offsetTail, alignNum) * alignNum;
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T1, AscendC::MicroAPI::RegTraitNumTwo> value0, value1, valueResult;
        AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> level0, level1, indexResult, helpIndex, indexMulResult, indexAddResult;

        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg tailMask = AscendC::MicroAPI::UpdateMask<uint32_t>(tailSize);
        AscendC::MicroAPI::MaskReg oneMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg cmpMask;

        AscendC::MicroAPI::Duplicate(helpIndex, VALUE_ONE, fullMask);
        vdup((AscendC::MicroAPI::RegTensor<uint32_t>&)helpIndex.reg[0], VL, oneMask, MODE_MERGING);

        // 用于加法的寄存器
        AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::MicroAPI::Duplicate(vregIndicesAdd1, VALUE_ONE, fullMask);
        AscendC::MicroAPI::Duplicate(vregIndicesAddStartR, startR, fullMask);
        
        // 主块, 载入i片数据，初始化i片索引
        AscendC::MicroAPI::UnalignReg uValue, uIndex, uSrc;
        AscendC::MicroAPI::DataCopyUnAlignPre(uSrc, (__local_mem__ T1*&)src);
        for (uint16_t a = 0; a < dimA; a++) {
            AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(value0, uSrc, (__local_mem__ T1*&)src, offset1);
            AscendC::MicroAPI::Duplicate(level0, VALUE_ZERO, fullMask);
            AscendC::MicroAPI::Duplicate(level1, VALUE_ZERO, fullMask);
            // 载入i+1片数据，初始化i+1片索引，比较 i and i+1
            for (uint16_t r = 0; r < loopTime; r++) {
                AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(value1, uSrc, (__local_mem__ T1*&)src, VL);
                // 微指令 AscendC::MicroAPI::Duplicate(level1, r + 1, fullMask);
                Add(level1, level1, vregIndicesAdd1, fullMask);
                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T1, CMPMODE::LE>(cmpMask, value0, value1, fullMask);
                } else {
                    AscendC::MicroAPI::Compare<T1, CMPMODE::GE>(cmpMask, value0, value1, fullMask);
                }
                AscendC::MicroAPI::Select(value0, value0, value1, cmpMask);
                AscendC::MicroAPI::Select(level0, level0, level1, cmpMask);
            }

            // 载入尾快，与主块比较
            for (uint16_t r = 0; r < tailLoop; r++) {
                AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(value1, uSrc, (__local_mem__ T1*&)src, offsetTail);
                AscendC::MicroAPI::Duplicate(level1, mainLoop, fullMask);
                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T1, CMPMODE::LT>(cmpMask, value1, value0, tailMask);
                } else {
                    AscendC::MicroAPI::Compare<T1, CMPMODE::GT>(cmpMask, value1, value0, tailMask);
                }
                AscendC::MicroAPI::Select(value0, value1, value0, cmpMask);
                AscendC::MicroAPI::Select(level0, level1, level0, cmpMask);
            }

            // 输出最终值 + 相对索引
            if constexpr (isMin) {
                AscendC::MicroAPI::ReduceMin(valueResult, value0, fullMask);
            } else {
                AscendC::MicroAPI::ReduceMax(valueResult, value0, fullMask);
            }
            AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__local_mem__ T1*&)valueDst, valueResult, uValue, 1);
            // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
            AscendC::MicroAPI::Duplicate(valueResult, valueResult, fullMask);
            AscendC::MicroAPI::Compare<T1, CMPMODE::EQ>(cmpMask, valueResult, value0, fullMask);
            AscendC::MicroAPI::ReduceMin(indexResult, level0, cmpMask);
            AscendC::MicroAPI::Mul(indexMulResult, helpIndex, indexResult, fullMask);
            AscendC::MicroAPI::DeInterleave(indexResult, indexAddResult, indexMulResult, helpIndex);
            Add(indexResult, indexResult, indexAddResult, oneMask);
            Add(indexResult, indexResult, vregIndicesAddStartR, oneMask);
            AscendC::MicroAPI::DataCopyUnAlign<T2, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__local_mem__ T2*&)indexDst, (AscendC::MicroAPI::RegTensor<T2> &)indexResult, uIndex, 1);
        }
        // 非对齐搬出后处理
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T1*&)valueDst, uValue, 0);
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T2*&)indexDst, uIndex, 0);
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxV1Int64ForRLessThanVL64(__local_mem__ T1 *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ T1 *src, uint16_t dimA, uint16_t dimR, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint32_t offset1 = dimR;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        offset1 = CeilDivision(offset1, alignNum) * alignNum;
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T1, AscendC::MicroAPI::RegTraitNumTwo> value0, valueResult;
        AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> level0, indexResult, helpIndex, indexMulResult, indexAddResult;

        uint32_t tailSize = dimR;
        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg tailMask = AscendC::MicroAPI::UpdateMask<uint32_t>(tailSize);
        AscendC::MicroAPI::MaskReg oneMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg cmpMask;

        AscendC::MicroAPI::Duplicate(helpIndex, VALUE_ONE, fullMask);
        vdup((AscendC::MicroAPI::RegTensor<uint32_t>&)helpIndex.reg[0], VL, oneMask, MODE_MERGING);

        // 用于加法的寄存器
        AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::MicroAPI::Duplicate(vregIndicesAdd1, 1, fullMask);
        AscendC::MicroAPI::Duplicate(vregIndicesAddStartR, startR, fullMask);
        
        // 主块, 载入i片数据，初始化i片索引
        AscendC::MicroAPI::UnalignReg uValue, uIndex, uSrc;
        AscendC::MicroAPI::DataCopyUnAlignPre(uSrc, (__local_mem__ T1*&)src);
        for (uint16_t a = 0; a < dimA; a++) {
            AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(value0, uSrc, (__local_mem__ T1*&)src, offset1);
            vdup((AscendC::MicroAPI::RegTensor<uint32_t>&)level0.reg[0], 0, tailMask, MODE_ZEROING);
            vdup((AscendC::MicroAPI::RegTensor<uint32_t>&)level0.reg[1], 0, tailMask, MODE_ZEROING);

            // 输出最终值 + 相对索引
            if constexpr (isMin) {
                AscendC::MicroAPI::ReduceMin(valueResult, value0, tailMask);
            } else {
                AscendC::MicroAPI::ReduceMax(valueResult, value0, tailMask);
            }
            AscendC::MicroAPI::DataCopyUnAlign<T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__local_mem__ T1*&)valueDst, valueResult, uValue, 1);
            // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
            AscendC::MicroAPI::Duplicate(valueResult, valueResult, tailMask);
            AscendC::MicroAPI::Compare<T1, CMPMODE::EQ>(cmpMask, valueResult, value0, tailMask);
            AscendC::MicroAPI::ReduceMin(indexResult, level0, cmpMask);
            AscendC::MicroAPI::Mul(indexMulResult, helpIndex, indexResult, fullMask);
            AscendC::MicroAPI::DeInterleave(indexResult, indexAddResult, indexMulResult, helpIndex);
            Add(indexResult, indexResult, indexAddResult, oneMask);
            Add(indexResult, indexResult, vregIndicesAddStartR, oneMask);
            AscendC::MicroAPI::DataCopyUnAlign<T2, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__local_mem__ T2*&)indexDst, (AscendC::MicroAPI::RegTensor<T2> &)indexResult, uIndex, 1);
        }
        // 非对齐搬出后处理
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T1*&)valueDst, uValue, 0);
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T2*&)indexDst, uIndex, 0);
    }
}

// RA场景VF接口，R < VL && A >= coreNum * VL
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxRaInt64(__local_mem__ void *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ void *src, uint16_t dimA, uint16_t alignA, uint16_t dimR,
    uint32_t startR)
{
    uint16_t VL = GetVLSize();
    // 向上对齐算上尾块次数
    uint16_t loopA = CeilDivision(dimA, VL);
    uint32_t processA = dimA;
    uint32_t sregMask = dimA;
    uint16_t loopR = dimR > 1 ? (dimR - 1) : 0;
    uint16_t offsetA = dimA >= VL ? VL : dimA;
    
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<T4, AscendC::MicroAPI::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::MicroAPI::RegTensor<T5, AscendC::MicroAPI::RegTraitNumTwo> vregIndices0, vregIndices1;

        AscendC::MicroAPI::RegTensor<T5, AscendC::MicroAPI::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::MicroAPI::Duplicate(vregIndicesAdd1, VALUE_ONE, fullMask);
        AscendC::MicroAPI::Duplicate(vregIndicesAddStartR, startR, fullMask);

        AscendC::MicroAPI::UnalignReg uSrc,tSrc;

        AscendC::MicroAPI::MaskReg mask, cmpMask;

        for (uint16_t i = 0; i < loopA; i++) {
            auto tmpPtr = (__local_mem__ T4*&)src + alignA;
            mask = AscendC::MicroAPI::UpdateMask<uint32_t>(processA);
            AscendC::MicroAPI::DataCopyUnAlignPre(uSrc, (__local_mem__ T4*&)src);
            AscendC::MicroAPI::DataCopyUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregValue0, uSrc,
                (__local_mem__ T4*&)src, offsetA);
            AscendC::MicroAPI::Duplicate(vregIndices0, VALUE_ZERO, fullMask);
            AscendC::MicroAPI::Duplicate(vregIndices1, VALUE_ZERO, fullMask);

            // 从第二行的R开始计算比较
            for (uint16_t j = 0; j < loopR; j++) {
                Add(vregIndices1, vregIndices1, vregIndicesAdd1, mask);
                AscendC::MicroAPI::DataCopyUnAlignPre(tSrc, (__local_mem__ T4*&)tmpPtr);
                AscendC::MicroAPI::DataCopyUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vregValue1, tSrc, (__local_mem__ T4*&)tmpPtr, VL);
                tmpPtr += (alignA - VL);
                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                AscendC::MicroAPI::Select(vregValue0, vregValue0, vregValue1, cmpMask);
                AscendC::MicroAPI::Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            
            DataCopy((__local_mem__ T4*)valueDst + i * VL, vregValue0, mask);

            Add(vregIndices0, vregIndices0, vregIndicesAddStartR, mask);
            DataCopy((__local_mem__ T5*)indexDst + i * VL, vregIndices0, mask);
        }
    }
}

// AR场景Gather接口，R < VL && A >= coreNum * VL
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherRaInt64(__local_mem__ void *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ void *src, uint16_t dimA, uint16_t dimR)
{
    uint16_t VL = GetVLSize();
    // 向上对齐算上尾块次数
    uint16_t loopA = CeilDivision(dimA, VL);
    uint32_t processA = dimA;
    uint16_t loopR = dimR > 1 ? (dimR - 1) : 0;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T4, AscendC::MicroAPI::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::MicroAPI::RegTensor<T5, AscendC::MicroAPI::RegTraitNumTwo> vregIndices0, vregIndices1;
        AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo> idx, idxInit;
        AscendC::MicroAPI::MaskReg mask, cmpMask;
        mask = AscendC::MicroAPI::UpdateMask<uint32_t>(processA);

        for (uint16_t i = 0; i < loopA; i++) {
            // Gather实现UB内转置
            uint32_t startIdx = i * VL;
            // 构造起始为 i * VL的1递增数列实现AR转RA
            Arange(idxInit, startIdx);
            Muls(idxInit, idxInit, dimR, mask);
            // Gather的偏移量索引要用uint类型
            DataCopyGather(vregValue0, (__local_mem__ T4*&)src, (AscendC::MicroAPI::RegTensor<uint32_t> &)idxInit, mask);
            // 初始化索引
            Duplicate(vregIndices0, 0);
            for (uint16_t j = 0; j < loopR; j++) {
                Duplicate(vregIndices1, j + 1, mask);
                // 更新下VL个数的索引
                Adds(idx, idxInit, (j + 1), mask);
                DataCopyGather(vregValue1, (__local_mem__ T4*&)src, (AscendC::MicroAPI::RegTensor<uint32_t> &)idx, mask);
                if constexpr(isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            DataCopy((__local_mem__ T4*&)valueDst + i * VL, vregValue0, mask);
            DataCopy((__local_mem__ T5*&)indexDst + i * VL, vregIndices0, mask);
        }
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6, bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherV1Int64(__local_mem__ void *valueDst,
    __local_mem__ void *indexDst, __local_mem__ void *src, uint16_t dimA, uint16_t dimR, uint16_t dimNextA,
    uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint16_t mainLoop = dimR / VL;
    uint16_t tailSize = dimR - mainLoop * VL;
    uint16_t loopTime = mainLoop > 1 ? mainLoop - 1 : 0;
    uint16_t tailLoop = (tailSize > 0 && mainLoop > 0) ? 1 : 0;
    uint32_t sreg1 = 1;
    uint32_t sreg2 = 2;
    uint32_t sregAdd = 1;
    uint32_t rNextA = dimR * dimNextA;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        rNextA = CeilDivision(rNextA, alignNum) * alignNum;
    }
    uint32_t offsetR = VL * dimNextA;
    uint32_t sregTail = dimR >= VL ? VL : dimR;
    uint32_t seg3 = tailSize;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T6, AscendC::MicroAPI::RegTraitNumTwo> initIndex, aIndex, rIndex, nextAIndex;
        AscendC::MicroAPI::RegTensor<T4, AscendC::MicroAPI::RegTraitNumTwo> value0, value1, valueResult;
        AscendC::MicroAPI::RegTensor<T6, AscendC::MicroAPI::RegTraitNumTwo> indexResult32, level0, level1, indexResult, helpIndex;
        AscendC::MicroAPI::UnalignReg uValue, uIndex;
        
        AscendC::MicroAPI::MaskReg oneMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg allMask32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg newMask = AscendC::MicroAPI::UpdateMask<uint32_t>(seg3);
        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::UpdateMask<uint32_t>(sregTail);
        AscendC::MicroAPI::MaskReg cmpMask;

        Duplicate(helpIndex, 1, fullMask);
        Duplicate<T6, AscendC::MicroAPI::MaskMergeMode::MERGING, T6>(helpIndex, VL, oneMask);
        Arange(initIndex, 0);                            // (0, 1, 2, ... , 63)
        Muls(initIndex, initIndex, dimNextA, allMask32); // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA)
        for (uint16_t a1 = 0; a1 < dimA; a1++) {
            // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA) + a * rNextA 只修改起始位置
            Adds(aIndex, initIndex, a1 * rNextA, allMask32);
            for (uint16_t a2 = 0; a2 < dimNextA; a2++) {
                Adds(nextAIndex, aIndex, a2, allMask32);
                DataCopyGather(value0, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<uint32_t> &)nextAIndex, fullMask);
                Duplicate(level0, 0, fullMask);
                // 载入i+1片数据，初始化i+1片索引，比较 i and i+1
                for (uint16_t r = 0; r < loopTime; r++) {
                    Adds(rIndex, nextAIndex, (r + 1) * offsetR, allMask32);
                    DataCopyGather(value1, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<uint32_t> &)rIndex, fullMask);
                    Duplicate(level1, r + 1, fullMask);
                    // NAN
                    if constexpr (isMin) {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, fullMask);
                    } else {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, fullMask);
                    }
                    Select(value0, value0, value1, cmpMask);
                    Select(level0, level0, level1, cmpMask);
                }
                // 载入尾块，与主块比较
                for (uint16_t r = 0; r < tailLoop; r++) {
                    Adds(rIndex, nextAIndex, mainLoop * offsetR, allMask32);
                    DataCopyGather(value1, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<uint32_t> &)rIndex, newMask);
                    Duplicate(level1, mainLoop, fullMask);
                    // NAN
                    if constexpr (isMin) {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::LT>(cmpMask, value1, value0, newMask);
                    } else {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::GT>(cmpMask, value1, value0, newMask);
                    }
                    Select(value0, value1, value0, cmpMask);
                    Select(level0, level1, level0, cmpMask);
                }
                // 输出最终值 + 相对索引
                if constexpr (isMin) {
                    ReduceMin(valueResult, value0, fullMask);
                } else {
                    ReduceMax(valueResult, value0, fullMask);
                }
                AscendC::MicroAPI::DataCopyUnAlign<T4, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__local_mem__ T4 *&)valueDst, valueResult, uValue, 1);
                // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
                Duplicate(valueResult, valueResult, fullMask);
                AscendC::MicroAPI::Compare<T4, CMPMODE::EQ>(cmpMask, valueResult, value0, fullMask);
                ReduceMin(indexResult, level0, cmpMask);
                Mul(indexResult, helpIndex, indexResult, allMask32);
                AscendC::MicroAPI::DeInterleave(indexResult, indexResult32, indexResult, helpIndex);
                Add(indexResult, indexResult, indexResult32, oneMask);
                Adds(indexResult32, indexResult, startR, oneMask);
                AscendC::MicroAPI::DataCopyUnAlign<T2, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__local_mem__ T2 *&)indexDst, (AscendC::MicroAPI::RegTensor<T6> &)indexResult32, uIndex, 1);
            }
        }
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T4 *&)valueDst, uValue, 0);
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T2 *&)indexDst, uIndex, 0);
    }
}
#else
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxV1(__local_mem__ void *valueDst,
    __local_mem__ void *indexDst, __local_mem__ void *src, uint16_t dimA, uint16_t dimR, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint16_t mainLoop = dimR / VL;
    uint16_t tailSize = dimR - mainLoop * VL;
    uint16_t loopTime = mainLoop > 1 ? mainLoop - 1 : 0;
    uint16_t tailLoop = (tailSize > 0 && mainLoop > 0) ? 1 : 0;
    uint32_t offset1 = dimR >= VL ? VL : dimR;
    uint32_t sreg0 = tailSize;
    uint32_t sreg1 = 1;
    uint32_t sreg2 = 2;
    uint32_t sreg3 = offset1;
    uint32_t sregAdd = 1;
    uint32_t offsetTail = tailSize;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        offset1 = CeilDivision(offset1, alignNum) * alignNum;
        offsetTail = CeilDivision(offsetTail, alignNum) * alignNum;
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg helpV0Mask, helpV1Mask, b8Mask, fullMask, tailMask, cmpMask, oneMask, twoMask;
        AscendC::MicroAPI::RegTensor<T4> value0, value1, valueResult;
        AscendC::MicroAPI::RegTensor<uint32_t> indexResult32;
        AscendC::MicroAPI::UnalignReg uValue, uIndex, uSrc;
        b8Mask = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg addMask = AscendC::MicroAPI::UpdateMask<uint32_t>(sregAdd);

        AscendC::MicroAPI::RegTensor<T5> level0, level1, indexResult, helpIndex;
        fullMask = AscendC::MicroAPI::UpdateMask<T5>(sreg3);
        oneMask = AscendC::MicroAPI::UpdateMask<T5>(sreg1);
        twoMask = AscendC::MicroAPI::UpdateMask<T5>(sreg2);
        tailMask = AscendC::MicroAPI::UpdateMask<T5>(sreg0);

        Duplicate(helpIndex, VALUE_ONE, fullMask);
        Duplicate<T5, AscendC::MicroAPI::MaskMergeMode::MERGING, T5>(helpIndex, VL, oneMask);
        AscendC::MicroAPI::DataCopyUnAlignPre(uSrc, (__local_mem__ T4 *&)src);
        for (uint16_t a = 0; a < dimA; a++) {
            AscendC::MicroAPI::DataCopyUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(value0, uSrc,
                (__local_mem__ T4 *&)src, offset1);
            Duplicate(level0, 0, fullMask);
            // 载入i+1片数据，初始化i+1片索引，比较 i and i+1
            for (uint16_t r = 0; r < loopTime; r++) {
                AscendC::MicroAPI::DataCopyUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(value1, uSrc,
                    (__local_mem__ T4 *&)src, VL);
                Duplicate(level1, r + 1, fullMask);
                // NAN
                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, fullMask);
                } else {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, fullMask);
                }
                AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, fullMask);
                AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, helpV0Mask, b8Mask);
                Select(value0, value0, value1, cmpMask);
                Select(level0, level0, level1, cmpMask);
            }
            // 载入尾块，与主块比较
            for (uint16_t r = 0; r < tailLoop; r++) {
                AscendC::MicroAPI::DataCopyUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(value1, uSrc,
                    (__local_mem__ T4 *&)src, offsetTail);
                Duplicate(level1, mainLoop, fullMask);
                // NAN
                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::LT>(cmpMask, value1, value0, tailMask);
                } else {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::GT>(cmpMask, value1, value0, tailMask);
                }
                AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, fullMask);
                AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV1Mask, value1, value1, tailMask);
                AscendC::MicroAPI::MaskNot(helpV0Mask, helpV0Mask, b8Mask);
                AscendC::MicroAPI::MaskAnd(helpV0Mask, helpV0Mask, helpV1Mask, b8Mask);
                AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, helpV0Mask, b8Mask);
                Select(value0, value1, value0, cmpMask);
                Select(level0, level1, level0, cmpMask);
            }
            // 输出最终值 + 相对索引
            if constexpr (isMin) {
                ReduceMin(valueResult, value0, fullMask);
            } else {
                ReduceMax(valueResult, value0, fullMask);
            }
            AscendC::MicroAPI::DataCopyUnAlign<T4, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__local_mem__ T4 *&)valueDst, valueResult, uValue, 1);
            // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
            Duplicate<T4, AscendC::MicroAPI::HighLowPart::LOWEST, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                valueResult, valueResult, fullMask);
            AscendC::MicroAPI::Compare<T4, CMPMODE::EQ>(cmpMask, valueResult, value0, fullMask);
            // NAN
            AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV1Mask, value0, value0, fullMask);
            AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, helpV1Mask, b8Mask);
            ReduceMin(indexResult, level0, cmpMask);
            Mul(indexResult, helpIndex, indexResult, twoMask);
            ReduceSum(indexResult32, indexResult, twoMask);
            Adds(indexResult32, indexResult32, startR, addMask);
            AscendC::MicroAPI::DataCopyUnAlign<T2, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__local_mem__ T2 *&)indexDst, (AscendC::MicroAPI::RegTensor<int32_t> &)indexResult32, uIndex, 1);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T4 *&)valueDst, uValue, 0);
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T2 *&)indexDst, uIndex, 0);
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6, bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherV1(__local_mem__ void *valueDst,
    __local_mem__ void *indexDst, __local_mem__ void *src, uint16_t dimA, uint16_t dimR, uint16_t dimNextA,
    uint32_t startR)
{
    uint16_t VL = GetVLSize();
    if (dimR < VL) {
        ArgMaxGatherV1ForRLessThanVL64<T4, T5, T6, needAlign>(valueDst, indexDst, src, dimA, dimR, dimNextA, startR);
        return;
    }
    uint16_t mainLoop = dimR / VL;
    uint32_t tailSize = dimR - mainLoop * VL;
    uint16_t loopTime = mainLoop - 1;
    uint16_t tailLoop = tailSize > 0 ? 1 : 0;
    uint32_t rNextA = dimR * dimNextA;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        rNextA = CeilDivision(rNextA, alignNum) * alignNum;
    }
    uint32_t offsetR = VL * dimNextA;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T6> initIndex, aIndex, rIndex, nextAIndex;
        AscendC::MicroAPI::RegTensor<T4> value0, value1, valueResult;
        AscendC::MicroAPI::RegTensor<uint32_t> indexResult32;
        AscendC::MicroAPI::UnalignReg uValue, uIndex, uSrc;
        AscendC::MicroAPI::RegTensor<T5> level0, level1, indexResult, helpIndex;

        AscendC::MicroAPI::MaskReg helpV0Mask, helpV1Mask, cmpMask;
        AscendC::MicroAPI::MaskReg oneMask = AscendC::MicroAPI::CreateMask<T5, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg twoMask = AscendC::MicroAPI::CreateMask<T5, AscendC::MicroAPI::MaskPattern::VL2>();
        AscendC::MicroAPI::MaskReg allMask32 = AscendC::MicroAPI::CreateMask<T5, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg newMask = AscendC::MicroAPI::UpdateMask<T5>(tailSize);
        AscendC::MicroAPI::MaskReg addMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg b8Mask = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

        Duplicate(helpIndex, VALUE_ONE, allMask32);
        Duplicate<T5, AscendC::MicroAPI::MaskMergeMode::MERGING, T5>(helpIndex, VL, oneMask);
        Arange(initIndex, 0);                            // (0, 1, 2, ... , 63)
        Muls(initIndex, initIndex, dimNextA, allMask32); // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA)
        for (uint16_t a1 = 0; a1 < dimA; a1++) {
            // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA) + a * rNextA 只修改起始位置
            Adds(aIndex, initIndex, a1 * rNextA, allMask32);
            for (uint16_t a2 = 0; a2 < dimNextA; a2++) {
                Adds(nextAIndex, aIndex, a2, allMask32);
                DataCopyGather(value0, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<T5> &)nextAIndex,
                    allMask32);
                Duplicate(level0, VALUE_ZERO, allMask32);
                // 载入i+1片数据，初始化i+1片索引，比较 i and i+1
                for (uint16_t r = 0; r < loopTime; r++) {
                    Adds(rIndex, nextAIndex, (r + 1) * offsetR, allMask32);
                    DataCopyGather(value1, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<T5> &)rIndex,
                        allMask32);
                    Duplicate(level1, r + 1, allMask32);
                    // NAN
                    if constexpr (isMin) {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, allMask32);
                    } else {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, allMask32);
                    }
                    AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, allMask32);
                    AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, helpV0Mask, b8Mask);
                    Select(value0, value0, value1, cmpMask);
                    Select(level0, level0, level1, cmpMask);
                }
                // 载入尾块，与主块比较
                for (uint16_t r = 0; r < tailLoop; r++) {
                    Adds(rIndex, nextAIndex, mainLoop * offsetR, allMask32);
                    DataCopyGather(value1, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<T5> &)rIndex,
                        newMask);
                    Duplicate(level1, mainLoop, allMask32);
                    // NAN
                    if constexpr (isMin) {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::LT>(cmpMask, value1, value0, newMask);
                    } else {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::GT>(cmpMask, value1, value0, newMask);
                    }
                    AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, allMask32);
                    AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV1Mask, value1, value1, newMask);
                    AscendC::MicroAPI::MaskNot(helpV0Mask, helpV0Mask, b8Mask);
                    AscendC::MicroAPI::MaskAnd(helpV0Mask, helpV0Mask, helpV1Mask, b8Mask);
                    AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, helpV0Mask, b8Mask);
                    Select(value0, value1, value0, cmpMask);
                    Select(level0, level1, level0, cmpMask);
                }
                // 输出最终值 + 相对索引
                if constexpr (isMin) {
                    ReduceMin(valueResult, value0, allMask32);
                } else {
                    ReduceMax(valueResult, value0, allMask32);
                }
                AscendC::MicroAPI::DataCopyUnAlign<T4, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__local_mem__ T4 *&)valueDst, valueResult, uValue, 1);
                // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
                Duplicate<T4, AscendC::MicroAPI::HighLowPart::LOWEST, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    valueResult, valueResult, allMask32);
                AscendC::MicroAPI::Compare<T4, CMPMODE::EQ>(cmpMask, valueResult, value0, allMask32);
                // NAN
                AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV1Mask, value0, value0, allMask32);
                AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, helpV1Mask, b8Mask);
                ReduceMin(indexResult, level0, cmpMask);
                Mul(indexResult, helpIndex, indexResult, twoMask);
                ReduceSum(indexResult32, indexResult, twoMask);
                Adds(indexResult32, indexResult32, startR, addMask);
                AscendC::MicroAPI::DataCopyUnAlign<T2, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__local_mem__ T2 *&)indexDst, (AscendC::MicroAPI::RegTensor<int32_t> &)indexResult32, uIndex, 1);
            }
        }
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T4 *&)valueDst, uValue, 0);
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T2 *&)indexDst, uIndex, 0);
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6, bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherV1ForRLessThanVL64(
    __local_mem__ void *valueDst, __local_mem__ void *indexDst, __local_mem__ void *src, uint16_t dimA, 
    uint16_t dimR, uint16_t dimNextA, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint32_t tailSize = dimR;
    uint32_t rNextA = dimR * dimNextA;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        rNextA = CeilDivision(rNextA, alignNum) * alignNum;
    }
    uint32_t offsetR = VL * dimNextA;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T6> initIndex, aIndex, rIndex, nextAIndex;
        AscendC::MicroAPI::RegTensor<T4> value0, value1, valueResult;
        AscendC::MicroAPI::RegTensor<uint32_t> indexResult32;
        AscendC::MicroAPI::UnalignReg uValue, uIndex, uSrc;
        AscendC::MicroAPI::RegTensor<T5> level0, level1, indexResult, helpIndex;

        AscendC::MicroAPI::MaskReg helpV1Mask, cmpMask;
        AscendC::MicroAPI::MaskReg oneMask = AscendC::MicroAPI::CreateMask<T5, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg twoMask = AscendC::MicroAPI::CreateMask<T5, AscendC::MicroAPI::MaskPattern::VL2>();
        AscendC::MicroAPI::MaskReg newMask = AscendC::MicroAPI::UpdateMask<T5>(tailSize);
        AscendC::MicroAPI::MaskReg allMask32 = AscendC::MicroAPI::CreateMask<T5, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg addMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg b8Mask = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

        Duplicate(helpIndex, VALUE_ONE, newMask);
        Duplicate<T5, AscendC::MicroAPI::MaskMergeMode::MERGING, T5>(helpIndex, VL, oneMask);
        Arange(initIndex, 0);                            // (0, 1, 2, ... , 63)
        Muls(initIndex, initIndex, dimNextA, allMask32); // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA)
        for (uint16_t a1 = 0; a1 < dimA; a1++) {
            // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA) + a * rNextA 只修改起始位置
            Adds(aIndex, initIndex, a1 * rNextA, allMask32);
            for (uint16_t a2 = 0; a2 < dimNextA; a2++) {
                Adds(nextAIndex, aIndex, a2, allMask32);
                DataCopyGather(value0, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<T5> &)nextAIndex,
                    newMask);
                Duplicate(level0, VALUE_ZERO, newMask);
                // 输出最终值 + 相对索引
                if constexpr (isMin) {
                    ReduceMin(valueResult, value0, newMask);
                } else {
                    ReduceMax(valueResult, value0, newMask);
                }
                AscendC::MicroAPI::DataCopyUnAlign<T4, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__local_mem__ T4 *&)valueDst, valueResult, uValue, 1);
                // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
                Duplicate<T4, AscendC::MicroAPI::HighLowPart::LOWEST, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    valueResult, valueResult, newMask);
                AscendC::MicroAPI::Compare<T4, CMPMODE::EQ>(cmpMask, valueResult, value0, newMask);
                // NAN
                AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV1Mask, value0, value0, newMask);
                AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, helpV1Mask, b8Mask);
                ReduceMin(indexResult, level0, cmpMask);
                Mul(indexResult, helpIndex, indexResult, twoMask);
                ReduceSum(indexResult32, indexResult, twoMask);
                Adds(indexResult32, indexResult32, startR, addMask);
                AscendC::MicroAPI::DataCopyUnAlign<T2, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__local_mem__ T2 *&)indexDst, (AscendC::MicroAPI::RegTensor<int32_t> &)indexResult32, uIndex, 1);
            }
        }
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T4 *&)valueDst, uValue, 0);
        AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ T2 *&)indexDst, uIndex, 0);
    }
}

#endif

// AR场景Gather接口，R < VL && A >= cornNum * VL
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherRa(__local_mem__ void *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ void *src, uint16_t dimA, uint16_t dimR)
{
    uint16_t VL = GetVLSize();
    // 向上对齐算上尾块次数
    uint16_t loopA = CeilDivision(dimA, VL);
    uint32_t processA = dimA;
    uint32_t sregMask = dimA;
    uint16_t loopR = dimR > 1 ? (dimR - 1) : 0;
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T4> vregValue0, vregValue1;
        AscendC::MicroAPI::RegTensor<T5> vregIndices0, vregIndices1;
        AscendC::MicroAPI::RegTensor<uint32_t> indicesLower, indicesHigher;
        AscendC::MicroAPI::RegTensor<T6> idx, idxInit;
        AscendC::MicroAPI::MaskReg mask, maskLower, maskHigher, cmpMask, nanMask;

        for (uint16_t i = 0; i < loopA; i++) {
            // Gather实现UB内转置
            mask = AscendC::MicroAPI::UpdateMask<T5>(processA);
            T6 startIdx = i * VL;
            // 构造起始为i * VL的1递增数列实现AR转RA
            Arange(idxInit, startIdx);
            Muls(idxInit, idxInit, dimR, mask);
            // Gather的偏移量索引要用uint类型
            DataCopyGather(vregValue0, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<T5> &)idxInit, mask);
            // 初始化索引
            Duplicate(vregIndices0, 0);
            for (uint16_t j = 0; j < loopR; j++) {
                Duplicate(vregIndices1, j + 1, mask);
                // 更新下VL个数的索引
                Adds(idx, idxInit, (j + 1), mask);
                DataCopyGather(vregValue1, (__local_mem__ T4 *&)src, (AscendC::MicroAPI::RegTensor<T5> &)idx, mask);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, mask);
                AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, nanMask, mask);
                // 获取最大value
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                // 获取最大value索引
                Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            AscendC::MicroAPI::AddrReg indicesOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(i, VL);
            AscendC::MicroAPI::AddrReg valueOffset = AscendC::MicroAPI::CreateAddrReg<T4>(i, VL);
            DataCopy((__local_mem__ T4 *&)valueDst, vregValue0, valueOffset, mask);
            if constexpr (IsSameType<T1, half>::value) {
                maskLower = AscendC::MicroAPI::UpdateMask<uint32_t>(sregMask);
                AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::LOWEST>(indicesLower, vregIndices0);
                AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::HIGHEST>(indicesHigher, vregIndices0);
                DataCopy(indexDst, (AscendC::MicroAPI::RegTensor<int32_t> &)indicesLower, indicesOffset, maskLower);
                maskHigher = AscendC::MicroAPI::UpdateMask<uint32_t>(sregMask);
                DataCopy(indexDst + repeatElmB32, (AscendC::MicroAPI::RegTensor<int32_t> &)indicesHigher, indicesOffset,
                    maskHigher);
            } else {
                DataCopy(indexDst, (AscendC::MicroAPI::RegTensor<int32_t> &)vregIndices0, indicesOffset, mask);
            }
        }
    }
}

// RA场景VF接口，R < VL && A >= cornNum * VL
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, bool isBfloat16Casted>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxRa(__local_mem__ void *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ void *src, uint16_t dimA, uint16_t alignA, uint16_t dimR,
    uint32_t startR)
{
    uint16_t VL = GetVLSize();
    if constexpr (IsSameType<T1, bfloat16_t>::value && !isBfloat16Casted) {
        VL = Ops::Base::GetVRegSize() / sizeof(T1);
    }
    // 向上对齐算上尾块次数
    uint16_t loopA = CeilDivision(dimA, VL);
    uint32_t processA = dimA;
    uint32_t sregMask = dimA;
    uint16_t loopR = dimR > 1 ? (dimR - 1) : 0;
    uint16_t offsetA = dimA >= VL ? VL : dimA;
    // B16偏移变量
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T4> vregValue0, vregValue1;
        // 指定传入uint类型作为索引
        AscendC::MicroAPI::RegTensor<T5> vregIndices0, vregIndices1;
        AscendC::MicroAPI::RegTensor<uint32_t> indicesLower, indicesHigher;
        AscendC::MicroAPI::UnalignReg uSrc,tSrc;
        AscendC::MicroAPI::MaskReg mask, maskLower, maskHigher, cmpMask, nanMask;
        for (uint16_t i = 0; i < loopA; i++) {
            auto tmpPtr = (__local_mem__ T4 *&)src + alignA;
            mask = AscendC::MicroAPI::UpdateMask<T5>(processA);
            AscendC::MicroAPI::DataCopyUnAlignPre(uSrc, (__local_mem__ T4 *&)src);
            AscendC::MicroAPI::DataCopyUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregValue0, uSrc,
                (__local_mem__ T4 *&)src, offsetA);
            // 初始化索引
            Duplicate(vregIndices0, 0);
            for (uint16_t j = 0; j < loopR; j++) { // 从第二行的R开始计算比较
                Duplicate(vregIndices1, j + 1, mask);
                AscendC::MicroAPI::DataCopyUnAlignPre(tSrc, (__local_mem__ T4 *&)tmpPtr);
                AscendC::MicroAPI::DataCopyUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregValue1,
                    tSrc, (__local_mem__ T4 *&)tmpPtr, VL);
            tmpPtr += (alignA - VL);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, mask);
                AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, nanMask, mask);
                // 获取最大value
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                // 获取最大value索引
                Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            // CreateAddrReg需要区分数据类型使用
            AscendC::MicroAPI::AddrReg indicesOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(i, VL);
            AscendC::MicroAPI::AddrReg valueOffset = AscendC::MicroAPI::CreateAddrReg<T4>(i, VL);
            DataCopy((__local_mem__ T4 *&)valueDst, vregValue0, valueOffset, mask);
            if constexpr (IsSameType<T1, half>::value || (!isBfloat16Casted && IsSameType<T1, bfloat16_t>::value)) {
                maskLower = AscendC::MicroAPI::UpdateMask<uint32_t>(sregMask);
                AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::LOWEST>(indicesLower, vregIndices0);
                AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::HIGHEST>(indicesHigher, vregIndices0);
                Adds(indicesLower, indicesLower, startR, maskLower);
                DataCopy(indexDst, (AscendC::MicroAPI::RegTensor<int32_t> &)indicesLower, indicesOffset, maskLower);
                maskHigher = AscendC::MicroAPI::UpdateMask<uint32_t>(sregMask);
                Adds(indicesHigher, indicesHigher, startR, maskHigher);
                DataCopy(indexDst + repeatElmB32, (AscendC::MicroAPI::RegTensor<int32_t> &)indicesHigher, indicesOffset,
                    maskHigher);
            } else {
                Adds(vregIndices0, vregIndices0, startR, mask);
                DataCopy(indexDst, (AscendC::MicroAPI::RegTensor<int32_t> &)vregIndices0, indicesOffset, mask);
            }
        }
    }
}


template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5>  // T4:Value, T5:与value位宽一致的uint类型可以是uint16 uint32 uint64三种
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxAra(__local_mem__ T4 *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ T4 *src, uint16_t dimA0, uint16_t dimR, uint16_t dimA1, uint32_t startR)
{
    uint16_t VL = Ops::Base::GetVRegSize() / sizeof(T4);
    uint16_t loopsA0 = dimA0;
    uint16_t loopsA1 = CeilDivision(dimA1, VL);
    uint16_t loopsR = dimR;
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T4> vregValue0, vregValue1;
        AscendC::MicroAPI::RegTensor<T5> vregIndices0, vregIndices1;
        AscendC::MicroAPI::RegTensor<uint32_t> indicesLower, indicesHigher;
        AscendC::MicroAPI::MaskReg maskReg, cmpMask, nanMask, maskLower, maskHigher;

        for (uint16_t loopA0 = 0; loopA0 < loopsA0; loopA0++) {
            uint32_t count = dimA1;
            uint32_t sregMask = dimA1;
            for (uint16_t loopA1 = 0; loopA1 < loopsA1; loopA1++) {
                maskReg = AscendC::MicroAPI::UpdateMask<T4>(count);
                AscendC::MicroAPI::DataCopy(vregValue0, src + loopA0 * dimR * dimA1 + loopA1 * VL);
                Duplicate(vregIndices0, 0);
                for (uint16_t loopR = 1; loopR < loopsR; loopR++) {
                    Duplicate(vregIndices1, loopR, maskReg);
                    AscendC::MicroAPI::DataCopy(vregValue1, src + loopA0 * dimR * dimA1 + loopR * dimA1 + loopA1 * VL);
                    if constexpr (isMin) {
                        Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, maskReg);
                    } else {
                        Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, maskReg);
                    }
                    Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, maskReg);  // 处理nan
                    AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, nanMask, maskReg);
                    // 获取最大value
                    Select(vregValue0, vregValue0, vregValue1, cmpMask);
                    // 获取最大value索引
                    Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
                }
                AscendC::MicroAPI::DataCopy(valueDst + loopA0 * dimA1 + loopA1 * VL, vregValue0, maskReg);

                if constexpr (IsSameType<T5, uint16_t>::value) {
                    maskLower = AscendC::MicroAPI::UpdateMask<uint32_t>(sregMask);
                    AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        indicesLower, vregIndices0);
                    AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::HIGHEST>(
                        indicesHigher, vregIndices0);
                    Adds(indicesLower, indicesLower, startR, maskLower);
                    DataCopy(indexDst + loopA0 * dimA1 + loopA1 * VL,
                        (AscendC::MicroAPI::RegTensor<T2> &)indicesLower,
                        maskLower);
                    maskHigher = AscendC::MicroAPI::UpdateMask<uint32_t>(sregMask);
                    Adds(indicesHigher, indicesHigher, startR, maskHigher);
                    DataCopy(indexDst + loopA0 * dimA1 + loopA1 * VL + repeatElmB32,
                        (AscendC::MicroAPI::RegTensor<T2> &)indicesHigher,
                        maskHigher);
                } else {
                    Adds(vregIndices0, vregIndices0, startR, maskReg);
                    AscendC::MicroAPI::DataCopy(indexDst + loopA0 * dimA1 + loopA1 * VL,
                        (AscendC::MicroAPI::RegTensor<T2> &)vregIndices0,
                        maskReg);
                }
            }
        }
    }
}


template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxAraInt64(__local_mem__ T4 *valueDst,
    __local_mem__ T2 *indexDst, __local_mem__ T4 *src, uint16_t dimA0, uint16_t dimR, uint16_t dimA1,
    uint32_t startR)
{
    uint16_t VL = GetVLSize();
    // 向上对齐算上尾块次数
    uint16_t loopA1 = CeilDivision(dimA1, VL);
    uint32_t sregMask = dimA1;
    uint16_t loopR = dimR;
    
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<T4, AscendC::MicroAPI::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::MicroAPI::RegTensor<T5, AscendC::MicroAPI::RegTraitNumTwo> vregIndices0, vregIndices1;
        AscendC::MicroAPI::RegTensor<T5, AscendC::MicroAPI::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::MicroAPI::UnalignReg uSrc;
        AscendC::MicroAPI::MaskReg mask, cmpMask;
        for (uint16_t loopA0 = 0; loopA0 < dimA0; loopA0++) {
            uint32_t processA = dimA1;
            AscendC::MicroAPI::Duplicate(vregIndicesAdd1, 1, fullMask);
            AscendC::MicroAPI::Duplicate(vregIndicesAddStartR, startR, fullMask);
            for (uint16_t i = 0; i < loopA1; i++) {
                mask = AscendC::MicroAPI::UpdateMask<uint32_t>(processA);
                AscendC::MicroAPI::DataCopy(vregValue0, src + loopA0 * dimR * dimA1 + i * VL);
                AscendC::MicroAPI::Duplicate(vregIndices0, 0, fullMask);
                AscendC::MicroAPI::Duplicate(vregIndices1, 0, fullMask);

                // 从第二行的R开始计算比较
                for (uint16_t j = 1; j < loopR; j++) {
                    Add(vregIndices1, vregIndices1, vregIndicesAdd1, mask);
                    AscendC::MicroAPI::DataCopy(vregValue1, src + loopA0 * dimR * dimA1 + j * dimA1 + i * VL);
                    if constexpr (isMin) {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                    } else {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                    }
                    AscendC::MicroAPI::Select(vregValue0, vregValue0, vregValue1, cmpMask);
                    AscendC::MicroAPI::Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
                }
                DataCopy((__local_mem__ T4*)valueDst + loopA0 * dimA1 + i * VL, vregValue0, mask);
                Add(vregIndices0, vregIndices0, vregIndicesAddStartR, mask);
                DataCopy((__local_mem__ T5*)indexDst + loopA0 * dimA1 + i * VL, vregIndices0, mask);
            }
        }
    }
}


template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, bool isBfloat16Casted>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::UpdateResult(__local_mem__ void *srcIndice,
    __local_mem__ void *srcValue, __local_mem__ void *dstIndice, __local_mem__ void *dstValue, uint64_t num)
{
    auto srcIndicePtr = (__local_mem__ T2 *)srcIndice;
    auto srcValuePtr = (__local_mem__ T4 *)srcValue;
    auto dstIndicePtr = (__local_mem__ T2 *)dstIndice;
    auto dstValuePtr = (__local_mem__ T4 *)dstValue;
    uint32_t sreg = static_cast<uint32_t>(num);
    uint32_t sreg32 = static_cast<uint32_t>(num);
    uint32_t vl = GetVLSize();
    if constexpr (IsSameType<T1, bfloat16_t>::value && !isBfloat16Casted) {
        vl = Ops::Base::GetVRegSize() / sizeof(T1);
    }
    uint16_t repeatTimes = CeilDivision(num, vl);
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);
    if constexpr (!IsSameType<T1, int64_t>::value) {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::UnalignReg ureg0;
            AscendC::MicroAPI::RegTensor<T4> value0, value1;
            AscendC::MicroAPI::RegTensor<T2> level0, level1, level2, level3;
            AscendC::MicroAPI::MaskReg preg, cmpMask, helpV0Mask, b8Mask, pregLower, pregHigher, preg32;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = AscendC::MicroAPI::UpdateMask<T4>(sreg);
                AscendC::MicroAPI::AddrReg valueOffset = AscendC::MicroAPI::CreateAddrReg<T4>(i, vl);
                AscendC::MicroAPI::AddrReg indicesOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(i, vl);
                DataCopy(value0, dstValuePtr, valueOffset);
                DataCopy(value1, srcValuePtr, valueOffset);
                // compare and select value
                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, preg);
                } else {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, preg);
                }
                AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, preg);
                AscendC::MicroAPI::MaskOr(cmpMask, cmpMask, helpV0Mask, b8Mask);
                Select(value0, value0, value1, cmpMask);
                DataCopy(dstValuePtr, value0, valueOffset, preg);
                if constexpr (IsSameType<T1, half>::value || (!isBfloat16Casted && IsSameType<T1, bfloat16_t>::value)) {
                    DataCopy(level0, dstIndicePtr, indicesOffset);
                    DataCopy(level1, srcIndicePtr, indicesOffset);
                    DataCopy(level2, dstIndicePtr + repeatElmB32, indicesOffset);
                    DataCopy(level3, srcIndicePtr + repeatElmB32, indicesOffset);
                    AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(pregLower, cmpMask);
                    AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(pregHigher, cmpMask);
                    Select(level0, level0, level1, pregLower);
                    Select(level2, level2, level3, pregHigher);
                    preg32 = AscendC::MicroAPI::UpdateMask<int32_t>(sreg32);
                    DataCopy(dstIndicePtr, level0, indicesOffset, preg32);
                    preg32 = AscendC::MicroAPI::UpdateMask<int32_t>(sreg32);
                    DataCopy(dstIndicePtr + repeatElmB32, level2, indicesOffset, preg32);
                } else {
                    DataCopy(level0, dstIndicePtr, indicesOffset);
                    DataCopy(level1, srcIndicePtr, indicesOffset);
                    Select(level0, level0, level1, cmpMask);
                    DataCopy(dstIndicePtr, level0, indicesOffset, preg);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T4, AscendC::MicroAPI::RegTraitNumTwo> value0, value1;
            AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> level0, level1;
            AscendC::MicroAPI::MaskReg preg, cmpMask, helpV0Mask;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = AscendC::MicroAPI::UpdateMask<uint32_t>(sreg);
                DataCopy(value0, (__local_mem__ T4 *)dstValuePtr + i * vl);
                DataCopy(value1, (__local_mem__ T4 *)srcValuePtr + i * vl);
                // compare and select value
                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, preg);
                } else {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, preg);
                }
                Select(value0, value0, value1, cmpMask);
                DataCopy((__local_mem__ T4 *)dstValuePtr + i * vl, value0, preg);

                DataCopy(level0, (__local_mem__ T2 *)dstIndicePtr + i * vl);
                DataCopy(level1, (__local_mem__ T2 *)srcIndicePtr + i * vl);
                Select(level0, level0, level1, cmpMask);
                DataCopy((__local_mem__ T2 *)dstIndicePtr + i * vl, level0, preg);
            }
        }
    }
}
} // namespace ArgMaxWithValue

#endif
// ARG_MAX_WITH_VALUE_BASE_H