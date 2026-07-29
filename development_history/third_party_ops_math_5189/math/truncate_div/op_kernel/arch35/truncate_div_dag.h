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
 * \file truncate_div_dag.h
 * \brief truncate_div dag
 */

#ifndef TRUNCATE_DIV_DAG_H
#define TRUNCATE_DIV_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/math_util.h"
#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#endif

namespace TruncateDivOp {
using namespace Ops::Base;
constexpr int TRUNCATE_DIV_CAST_MODE_NONE = 0;
constexpr int TRUNCATE_DIV_CAST_MODE_RINT = 1;
constexpr int8_t SAT_POS = 60;
constexpr int64_t INT64_MAX_VALUE = 9223372036854775807;
constexpr int64_t UINT32_MAX_VALUE = 4294967295;
const uint32_t UINT32_SIGN = 0x80000000;
const uint16_t UINT16_SIGN = 0x8000;
constexpr uint32_t TRUNCATE_DIV_SIMT_THREADS = 1024;

namespace TruncDag1 {
template <class T>
struct TruncCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline TruncCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask;
        if constexpr (std::is_same_v<T, float>) {
            MicroAPI::RegTensor<uint32_t, MicroAPI::RegTraitNumOne> vregOutInt;
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));

                    MicroAPI::Truncate<T, RoundMode::CAST_TRUNC, MicroAPI::MaskMergeMode::ZEROING>(
                        vregOutput, vregInput, mask);
                    MicroAPI::Duplicate(vregOutInt, UINT32_SIGN, mask);
                    MicroAPI::And(vregOutInt, vregOutInt, (MicroAPI::RegTensor<uint32_t>&)vregInput, mask);
                    MicroAPI::Or(vregOutInt, vregOutInt, (MicroAPI::RegTensor<uint32_t>&)vregOutput, mask);

                    // OpCopyOut
                    MicroAPI::DataCopy(
                        (__ubuf__ T*)(dstAddr + loopIdx * vlSize), (MicroAPI::RegTensor<T>&)vregOutInt, mask);
                }
            }
        } else {
            MicroAPI::RegTensor<uint16_t, MicroAPI::RegTraitNumOne> vregOutInt;
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));

                    MicroAPI::Truncate<T, RoundMode::CAST_TRUNC, MicroAPI::MaskMergeMode::ZEROING>(
                        vregOutput, vregInput, mask);
                    MicroAPI::Duplicate(vregOutInt, UINT16_SIGN, mask);
                    MicroAPI::And(vregOutInt, vregOutInt, (MicroAPI::RegTensor<uint16_t>&)vregInput, mask);
                    MicroAPI::Or(vregOutInt, vregOutInt, (MicroAPI::RegTensor<uint16_t>&)vregOutput, mask);

                    // OpCopyOut
                    MicroAPI::DataCopy(
                        (__ubuf__ T*)(dstAddr + loopIdx * vlSize), (MicroAPI::RegTensor<T>&)vregOutInt, mask);
                }
            }
        }
#endif
    }
};
} // namespace TruncDag1

template <class R, class T, int roundMode>
struct CastOverFlow : public Vec::ElemwiseUnaryOP<R, T> {
    __aicore__ inline CastOverFlow(LocalTensor<R>& dst, LocalTensor<T>& src, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        SetCtrlSpr<SAT_POS, SAT_POS>(0);
        constexpr static MicroAPI::CastTrait castTrait3 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_TRUNC};
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> vreg0;
            MicroAPI::RegTensor<R> vreg1;
            MicroAPI::MaskReg preg0;
            uint32_t size = count;
            uint16_t vfLoopNum = (size + (VECTOR_REG_WIDTH / sizeof(T)) - 1) / (VECTOR_REG_WIDTH / sizeof(T));
            __local_mem__ T* bufferIn0Addr = (__local_mem__ T*)src.GetPhyAddr();
            __local_mem__ R* bufferOut0Addr = (__local_mem__ R*)dst.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<T>(size);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                    vreg0, bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(T)));
                MicroAPI::Cast<R, T, castTrait3>(vreg1, vreg0, preg0);
                MicroAPI::DataCopy<R, MicroAPI::StoreDist::DIST_PACK_B16>(
                    bufferOut0Addr + i * (VECTOR_REG_WIDTH / sizeof(T)), vreg1, preg0);
            }
        }
        SetCtrlSpr<SAT_POS, SAT_POS>(1);
#endif
    }
};

template <class T>
struct TruncIntPostCompute : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline TruncIntPostCompute(
        const LocalTensor<T>& dst, const LocalTensor<T>& input1, const LocalTensor<T>& input2,
        const LocalTensor<T>& div, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T);
        __local_mem__ T* input1Addr = (__local_mem__ T*)input1.GetPhyAddr();
        __local_mem__ T* input2Addr = (__local_mem__ T*)input2.GetPhyAddr();
        __local_mem__ T* divAddr = (__local_mem__ T*)div.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> zeroValue;
            MicroAPI::RegTensor<T> defaultValue;
            MicroAPI::RegTensor<T> input1Value;
            MicroAPI::RegTensor<T> input2Value;
            MicroAPI::RegTensor<T> divValue;
            MicroAPI::RegTensor<T> resValue;
            MicroAPI::MaskReg preg;
            MicroAPI::MaskReg cmpValue;
            uint32_t sregMask = count;

            MicroAPI::Duplicate(zeroValue, T(0));
            MicroAPI::Duplicate(defaultValue, T(-1));

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = MicroAPI::UpdateMask<T>(sregMask);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(input2Value, input2Addr + VL_T * j);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(divValue, divAddr + VL_T * j);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(input1Value, input1Addr + VL_T * j);
                MicroAPI::Compare<T, CMPMODE::NE>(cmpValue, input2Value, zeroValue, preg);
                MicroAPI::Select(resValue, divValue, defaultValue, cmpValue);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
            }
        }
#endif
    }
};

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(TRUNCATE_DIV_SIMT_THREADS) inline void TruncDivInt_SIMT(
    __ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2, int count)
{
    for (uint32_t index = static_cast<uint32_t>(threadIdx.x); index < count;
         index += static_cast<uint32_t>(blockDim.x)) {
        bool pos_div_zero = ((src1[index] >= 0) && (src1[index] < INT64_MAX_VALUE) && (src2[index] == 0));
        bool div_zero = (src2[index] == 0);
        if (pos_div_zero) {
            dst[index] = UINT32_MAX_VALUE;
        } else if (div_zero) {
            dst[index] = -1;
        } else {
            dst[index] = src1[index] / src2[index];
        }
    }
}
#endif

template <class T>
struct TruncDivInt64 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline TruncDivInt64(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* dst_1 = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* src1_1 = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2_1 = (__ubuf__ T*)src2.GetPhyAddr();
        asc_vf_call<TruncDivInt_SIMT<T>>(dim3(TRUNCATE_DIV_SIMT_THREADS), dst_1, src1_1, src2_1, count);
#endif
    }
};

template <typename T1, typename T2, typename PromoteT>
struct TruncateDivFloatWithCast {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T2>, Placeholder::In1<T2>>;
    using CastX1 = Bind<Vec::Cast<PromoteT, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<PromoteT, T2, TRUNCATE_DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::DivHighPrecision<PromoteT>, CastX1, CastX2>;
    using TruncateValue = Bind<TruncDag1::TruncCustom<PromoteT>, DivValue>;
    using CastOut = Bind<Vec::Cast<PromoteT, PromoteT, TRUNCATE_DIV_CAST_MODE_RINT>, TruncateValue>;
    using OpCopyOut = Bind<Vec::CopyOut<PromoteT>, Placeholder::Out0<PromoteT>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// half and bfloat16
template <typename T1, typename PromoteT>
struct TruncateDivFloat16 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<PromoteT, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<PromoteT, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::DivHighPrecision<PromoteT>, CastX1, CastX2>;
    using TruncateValue = Bind<TruncDag1::TruncCustom<PromoteT>, DivValue>;
    using CastOut = Bind<Vec::Cast<T1, PromoteT, TRUNCATE_DIV_CAST_MODE_RINT>, TruncateValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct TruncateDivFloat {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivValue = Bind<Vec::DivHighPrecision<T1>, InputX1, InputX2>;
    using TruncateValue = Bind<TruncDag1::TruncCustom<T1>, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, TruncateValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2, typename PromoteT>
struct TruncateDivFloatWithCastScalar {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using CastX1 = Bind<Vec::Cast<PromoteT, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX1>;
    using DivValue = Bind<Vec::Muls<PromoteT>, CastX1, Placeholder::Var<float, 0>>;
    using TruncateValue = Bind<TruncDag1::TruncCustom<PromoteT>, DivValue>;
    using CastOut = Bind<Vec::Cast<PromoteT, PromoteT, TRUNCATE_DIV_CAST_MODE_RINT>, TruncateValue>;
    using OpCopyOut = Bind<Vec::CopyOut<PromoteT>, Placeholder::Out0<PromoteT>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename PromoteT>
struct TruncateDivFloat16Scalar {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using CastX1 = Bind<Vec::Cast<PromoteT, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX1>;
    using DivValue = Bind<Vec::Muls<PromoteT>, CastX1, Placeholder::Var<float, 0>>;
    using TruncateValue = Bind<TruncDag1::TruncCustom<PromoteT>, DivValue>;
    using CastOut = Bind<Vec::Cast<T1, PromoteT, TRUNCATE_DIV_CAST_MODE_RINT>, TruncateValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct TruncateDivFloatScalar {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using DivValue = Bind<Vec::Muls<T1>, InputX1, Placeholder::Var<float, 0>>;
    using TruncateValue = Bind<TruncDag1::TruncCustom<T1>, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, TruncateValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct TruncateDivIntS8 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;

    using CastX1 = Bind<Vec::Cast<T2, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<T2, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::Div<T2>, CastX1, CastX2>;
    using OpIntResult = Bind<CastOverFlow<T1, T2, TRUNCATE_DIV_CAST_MODE_RINT>, DivValue>;
    using ComputeValue = Bind<TruncIntPostCompute<T1>, InputX1, InputX2, OpIntResult>;

    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, ComputeValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct TruncateDivIntU8 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<T2, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<T2, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::Div<T2>, CastX1, CastX2>;
    using CastOut = Bind<Vec::Cast<T1, T2, TRUNCATE_DIV_CAST_MODE_NONE>, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct TruncateDivInt {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivValue = Bind<Vec::Div<T1>, InputX1, InputX2>;
    using ComputeValue = Bind<TruncIntPostCompute<T1>, InputX1, InputX2, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, ComputeValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct TruncateDivInt64 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivValue = Bind<TruncDivInt64<T1>, InputX1, InputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, DivValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2, typename PromoteT>
struct TruncateDivIntToFloat {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T2>, Placeholder::In1<T2>>;
    using CastX1 = Bind<Vec::Cast<PromoteT, T1, TRUNCATE_DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<PromoteT, T2, TRUNCATE_DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::DivHighPrecision<PromoteT>, CastX1, CastX2>;
    using TruncateValue = Bind<TruncDag1::TruncCustom<PromoteT>, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<PromoteT>, Placeholder::Out0<PromoteT>, TruncateValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2, typename PromoteT>
struct TruncateDivFloatToLowBit {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T2>, Placeholder::In1<T2>>;
    using CastX2 = Bind<Vec::Cast<PromoteT, T2, TRUNCATE_DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::DivHighPrecision<PromoteT>, InputX1, CastX2>;
    using TruncateValue = Bind<TruncDag1::TruncCustom<PromoteT>, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<PromoteT>, Placeholder::Out0<PromoteT>, TruncateValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace TruncateDivOp

#endif // TRUNCATE_DIV_DAG_H