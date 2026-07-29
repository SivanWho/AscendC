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
 * \file floor_mod_dag.h
 * \brief floor_mod dag
 */

#ifndef FLOOR_MOD_DAG_H
#define FLOOR_MOD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "op_kernel/math_util.h"
#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#endif

using namespace Ops::Base;
using namespace AscendC;

namespace FloorModOp {

constexpr int CAST_NONE_MODE = 0;

constexpr int FMOD_CMP_NE_MODE = 5;
constexpr int FMOD_SEL_TENSOR_TENSOR_MODE = 2;

constexpr uint32_t FMOD_B32_SIGN = 0X80000000;
constexpr uint64_t FMOD_B64_SIGN = 0x8000000000000000;

constexpr uint64_t FMOD_B64_MAX = 0Xffffffffffffffff;

#ifdef __CCE_AICORE__
constexpr static MicroAPI::CastTrait castTrait1 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

#endif

template <class T>
struct FmodPostCompute : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline FmodPostCompute(
        LocalTensor<T>& dst, LocalTensor<T>& fmodRes, LocalTensor<T>& inputX2, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T);
        __local_mem__ T* fmodResAddr = (__local_mem__ T*)fmodRes.GetPhyAddr();
        __local_mem__ T* inputX2Addr = (__local_mem__ T*)inputX2.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> zeroValue;
            MicroAPI::RegTensor<T> fmodResValue;
            MicroAPI::RegTensor<T> inputX2Value;
            MicroAPI::RegTensor<T> addValue;
            MicroAPI::RegTensor<T> resValue;

            MicroAPI::RegTensor<uint32_t> signValue;
            MicroAPI::RegTensor<uint32_t> fmodSignValue;
            MicroAPI::RegTensor<uint32_t> inputX2signValue;

            MicroAPI::MaskReg preg;
            MicroAPI::MaskReg negValue;
            MicroAPI::MaskReg signNegValue;
            MicroAPI::MaskReg resMaskValue;
            uint32_t sregMask = count;

            MicroAPI::Duplicate(zeroValue, T(0));
            MicroAPI::Duplicate(signValue, FMOD_B32_SIGN);

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = MicroAPI::UpdateMask<T>(sregMask);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(fmodResValue, fmodResAddr + VL_T * j);
                MicroAPI::Compare<T, CMPMODE::NE>(negValue, fmodResValue, zeroValue, preg);

                MicroAPI::And(fmodSignValue, (MicroAPI::RegTensor<uint32_t>&)fmodResValue, signValue, preg);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(inputX2Value, inputX2Addr + VL_T * j);
                MicroAPI::Add(addValue, fmodResValue, inputX2Value, preg);
                MicroAPI::And(inputX2signValue, (MicroAPI::RegTensor<uint32_t>&)inputX2Value, signValue, preg);
                MicroAPI::Compare<uint32_t, CMPMODE::NE>(signNegValue, fmodSignValue, inputX2signValue, preg);

                MicroAPI::MaskAnd(resMaskValue, signNegValue, negValue, preg);
                MicroAPI::Select(resValue, addValue, fmodResValue, resMaskValue);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
            }
        }
#endif
    }
};

template <class T1, class T2>
struct FmodCastFloatPostCompute : public Vec::ElemwiseBinaryOP<T1, T2, T2> {
    __aicore__ inline FmodCastFloatPostCompute(
        LocalTensor<T1>& dst, LocalTensor<T2>& fmodRes, LocalTensor<T2>& inputX2, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T2);
        __local_mem__ T2* fmodResAddr = (__local_mem__ T2*)fmodRes.GetPhyAddr();
        __local_mem__ T2* inputX2Addr = (__local_mem__ T2*)inputX2.GetPhyAddr();
        __local_mem__ T1* dstAddr = (__local_mem__ T1*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T2> zeroValue;
            MicroAPI::RegTensor<T2> fmodResValue;
            MicroAPI::RegTensor<T2> inputX2Value;
            MicroAPI::RegTensor<T2> addValue;
            MicroAPI::RegTensor<T2> resValue;
            MicroAPI::RegTensor<T1> resCastValue;

            MicroAPI::RegTensor<uint32_t> signValue;
            MicroAPI::RegTensor<uint32_t> fmodSignValue;
            MicroAPI::RegTensor<uint32_t> inputX2signValue;

            MicroAPI::MaskReg preg;
            MicroAPI::MaskReg negValue;
            MicroAPI::MaskReg signNegValue;
            MicroAPI::MaskReg resMaskValue;
            uint32_t sregMask = count;

            MicroAPI::Duplicate(zeroValue, T2(0));
            MicroAPI::Duplicate(signValue, FMOD_B32_SIGN);

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = MicroAPI::UpdateMask<T2>(sregMask);
                MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_NORM>(fmodResValue, fmodResAddr + VL_T * j);
                MicroAPI::Compare<T2, CMPMODE::NE>(negValue, fmodResValue, zeroValue, preg);

                MicroAPI::And(fmodSignValue, (MicroAPI::RegTensor<uint32_t>&)fmodResValue, signValue, preg);
                MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_NORM>(inputX2Value, inputX2Addr + VL_T * j);
                MicroAPI::Add(addValue, fmodResValue, inputX2Value, preg);
                MicroAPI::And(inputX2signValue, (MicroAPI::RegTensor<uint32_t>&)inputX2Value, signValue, preg);
                MicroAPI::Compare<uint32_t, CMPMODE::NE>(signNegValue, fmodSignValue, inputX2signValue, preg);

                MicroAPI::MaskAnd(resMaskValue, signNegValue, negValue, preg);
                MicroAPI::Select(resValue, addValue, fmodResValue, resMaskValue);
                MicroAPI::Cast<T1, float, castTrait1>(resCastValue, resValue, preg);
                MicroAPI::DataCopy<T1, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr + VL_T * j, resCastValue, preg);
            }
        }
#endif
    }
};

template <class T>
struct FmodIntPostCompute : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline FmodIntPostCompute(
        LocalTensor<T>& dst, LocalTensor<T>& input1, LocalTensor<T>& input2, LocalTensor<T>& div, const uint32_t& count)
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
            MicroAPI::RegTensor<T> signValue;
            MicroAPI::RegTensor<T> input1Value;
            MicroAPI::RegTensor<T> input2Value;
            MicroAPI::RegTensor<T> divValue;
            MicroAPI::RegTensor<T> mulValue;
            MicroAPI::RegTensor<T> subValue;
            MicroAPI::RegTensor<T> modValue;
            MicroAPI::RegTensor<T> modSignValue;
            MicroAPI::RegTensor<T> addValue;
            MicroAPI::RegTensor<T> input2SignValue;
            MicroAPI::RegTensor<T> resValue;

            MicroAPI::MaskReg preg;
            MicroAPI::MaskReg cmpValue;
            MicroAPI::MaskReg negValue;
            MicroAPI::MaskReg signNegValue;
            MicroAPI::MaskReg resMaskValue;
            uint32_t sregMask = count;

            MicroAPI::Duplicate(zeroValue, T(0));
            MicroAPI::Duplicate(defaultValue, T(-1));
            MicroAPI::Duplicate(signValue, FMOD_B32_SIGN);

            for (uint16_t j = 0; j < loopTimes; j++) {
                // handel -1
                preg = MicroAPI::UpdateMask<T>(sregMask);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(input2Value, input2Addr + VL_T * j);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(divValue, divAddr + VL_T * j);
                MicroAPI::Mul(mulValue, input2Value, divValue, preg);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(input1Value, input1Addr + VL_T * j);
                MicroAPI::Sub(subValue, input1Value, mulValue, preg);
                MicroAPI::Compare<T, CMPMODE::NE>(cmpValue, input2Value, zeroValue, preg);
                MicroAPI::Select(modValue, subValue, defaultValue, cmpValue);

                // post handel
                MicroAPI::Add(addValue, modValue, input2Value, preg);
                MicroAPI::Compare<T, CMPMODE::NE>(negValue, modValue, zeroValue, preg);
                MicroAPI::And(input2SignValue, input2Value, signValue, preg);
                MicroAPI::And(modSignValue, modValue, signValue, preg);
                MicroAPI::Compare<T, CMPMODE::NE>(signNegValue, modSignValue, input2SignValue, preg);
                MicroAPI::MaskAnd(resMaskValue, signNegValue, negValue, preg);
                MicroAPI::Select(resValue, addValue, modValue, resMaskValue);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
            }
        }
#endif
    }
};

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__
    LAUNCH_BOUND(1024) inline void FloorModInt_1(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2, int count)
{
    for (uint32_t index = static_cast<uint32_t>(threadIdx.x); index < count;
         index += static_cast<uint32_t>(blockDim.x)) {
        const auto rem = src1[index] % src2[index];
        bool signs_differ = ((rem < 0) != (src2[index] < 0));
        if (signs_differ && (rem != 0)) {
            dst[index] = rem + src2[index];
        } else {
            dst[index] = rem;
        }
    }
}
#endif

template <class T>
struct FloorModInt : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline FloorModInt(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* dst_1 = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* src1_1 = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2_1 = (__ubuf__ T*)src2.GetPhyAddr();
        asc_vf_call<FloorModInt_1<T>>(dim3(1024), dst_1, src1_1, src2_1, count);
#endif
    }
};

template <typename T>
struct FloorModFloatWithCastOp {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpCastX1 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX2>;
    using FmodRes = Bind<Vec::FmodHighPrecision<float>, OpCastX1, OpCastX2>;
    using Output = Bind<FmodCastFloatPostCompute<T, float>, FmodRes, OpCastX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Output>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct FloorModFloatOp {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using FmodRes = Bind<Vec::FmodHighPrecision<T>, OpInputX1, OpInputX2>;
    using Output = Bind<FmodPostCompute<T>, FmodRes, OpInputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Output>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct FloorModInt32Op {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using FmodRes = Bind<FloorModInt<T>, OpInputX1, OpInputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, FmodRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct FloorModInt64Op {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using FmodRes = Bind<FloorModInt<T>, OpInputX1, OpInputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, FmodRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace FloorModOp

#endif // FLOOR_MOD_DAG_H
