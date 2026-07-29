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
 * \file sign.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_SIGN_H
#define CANN_CUSTOM_OPS_SIGN_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace SignDag {
using namespace Ops::Base;
constexpr int COMPARE_MODE_GT = 1;
constexpr int COMPARE_MODE_LT = 0;
constexpr int CastModeBf16ToFp32 = 0;
constexpr int CastModeFp32ToBf16 = 1;

template <class T>
struct SignCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline SignCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t vlSize = VL;
        __VEC_SCOPE__
        {
            __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
            __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregInput;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregZeros;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregOnes;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregLeft;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregRight;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregOutput;

            AscendC::MicroAPI::MaskReg vregLeftMask;
            AscendC::MicroAPI::MaskReg vregRightMask;
            AscendC::MicroAPI::MaskReg mask;

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = AscendC::MicroAPI::UpdateMask<T, AscendC::MicroAPI::RegTraitNumOne>(count);
                // OpCopyIn0
                AscendC::MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                // compute sign
                AscendC::MicroAPI::CompareScalar<
                    T, AscendC::CMPMODE::GT, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>, T>(
                    vregLeftMask, vregInput, (T)0.0, mask);
                AscendC::MicroAPI::CompareScalar<
                    T, AscendC::CMPMODE::LT, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>, T>(
                    vregRightMask, vregInput, (T)0.0, mask);

                AscendC::MicroAPI::Duplicate(vregOnes, (T)1.0, mask);
                AscendC::MicroAPI::Duplicate(vregZeros, (T)0.0, mask);
                AscendC::MicroAPI::Select<T>(vregLeft, vregOnes, vregZeros, vregLeftMask);
                AscendC::MicroAPI::Select<T>(vregRight, vregOnes, vregZeros, vregRightMask);

                AscendC::MicroAPI::Sub(vregOutput, vregLeft, vregRight, mask);
                // OpCopyOut
                AscendC::MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};

template <class T>
struct SignCustomInt64 : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline SignCustomInt64(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH_2XVL / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t vlSize = VL;
        __VEC_SCOPE__
        {
            __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
            __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumTwo> vregInput;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumTwo> vregZeros;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumTwo> vregOnes;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumTwo> vregLeft;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumTwo> vregRight;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumTwo> vregOutput;

            AscendC::MicroAPI::MaskReg vregLeftMask;
            AscendC::MicroAPI::MaskReg vregRightMask;
            AscendC::MicroAPI::MaskReg mask;

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = AscendC::MicroAPI::UpdateMask<T, AscendC::MicroAPI::RegTraitNumTwo>(count);
                // OpCopyIn0
                AscendC::MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                // compute sign
                AscendC::MicroAPI::CompareScalar<
                    T, AscendC::CMPMODE::GT, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumTwo>, T>(
                    vregLeftMask, vregInput, (T)0.0, mask);
                AscendC::MicroAPI::CompareScalar<
                    T, AscendC::CMPMODE::LT, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumTwo>, T>(
                    vregRightMask, vregInput, (T)0.0, mask);

                AscendC::MicroAPI::Duplicate(vregOnes, (T)1.0, mask);
                AscendC::MicroAPI::Duplicate(vregZeros, (T)0.0, mask);
                AscendC::MicroAPI::Select<T>(vregLeft, vregOnes, vregZeros, vregLeftMask);
                AscendC::MicroAPI::Select<T>(vregRight, vregOnes, vregZeros, vregRightMask);

                AscendC::MicroAPI::Sub(vregOutput, vregLeft, vregRight, mask);
                // OpCopyOut
                AscendC::MicroAPI::DataCopy<int64_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                    (__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};

template <typename T>
struct SignForNumber {
    // 通过Compute构造计算图
    // y = compare(x > 0) - compare(x < 0)
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpSign = Bind<SignDag::SignCustom<T>, OpCopyIn>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSign>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    // 指定计算顺序
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct SignForInt64 {
    // 通过Compute构造计算图
    // y = compare(x > 0) - compare(x < 0)
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpSign = Bind<SignDag::SignCustomInt64<T>, OpCopyIn>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSign>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    // 指定计算顺序
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct SignForBf {
    // 通过Compute构造计算图
    // y = x if x is bool
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, CastModeBf16ToFp32>, OpCopyIn>;
    using OpSign = Bind<SignDag::SignCustom<float>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, float, CastModeFp32ToBf16>, OpSign>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}; // namespace SignDag

#endif // CANN_CUSTOM_OPS_SIGN_H