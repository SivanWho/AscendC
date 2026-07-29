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
 * \file log1p_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_LOG1P_DAG_H
#define CANN_CUSTOM_OPS_LOG1P_DAG_H
 
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace Log1pOp {
using namespace Ops::Base;
const float FP32_ONE = 1.0;
const float FP32_NEG_ONE = -1.0;
const float FP32_INF = (__builtin_inff());
#ifdef __CCE_AICORE__
constexpr static AscendC::MicroAPI::CastTrait castTrait0 = { AscendC::MicroAPI::RegLayout::ZERO,
AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
constexpr static AscendC::MicroAPI::CastTrait castTrait1 = { AscendC::MicroAPI::RegLayout::ZERO,
AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
#endif
namespace Log1pDag1 {
template<class T>
struct Log1pCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline Log1pCustom(LocalTensor<T> &dst, LocalTensor<T> &src, uint32_t count) {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint32_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputAddOne;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputMid;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg cmpMaskReg;
        if constexpr(std::is_same_v<T, float>) {
            __VEC_SCOPE__ {
                for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                    mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    
                    MicroAPI::Adds(vregInputAddOne, vregInput, FP32_ONE, mask);
                    MicroAPI::Adds(vregInputMid, vregInputAddOne, FP32_NEG_ONE, mask);
                    MicroAPI::Div(vregInputMid, vregInput, vregInputMid, mask);
                    MicroAPI::Log(vregOutput, vregInputAddOne, mask);
                    MicroAPI::Mul(vregOutput, vregOutput, vregInputMid, mask);
                    MicroAPI::CompareScalar<float, CMPMODE::NE>(cmpMaskReg, vregInputAddOne, FP32_ONE, mask);
                    MicroAPI::Select(vregOutput, vregOutput, vregInput, cmpMaskReg);
                    MicroAPI::CompareScalar<float, CMPMODE::NE>(cmpMaskReg, vregInputAddOne, FP32_INF, mask);
                    MicroAPI::Duplicate(vregInputMid, FP32_INF, mask);
                    MicroAPI::Select(vregOutput, vregOutput, vregInputMid, cmpMaskReg);

                    // OpCopyOut
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        } else {
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput16;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput16;
            __VEC_SCOPE__ {
                for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                    mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInput16, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);

                    MicroAPI::Adds(vregInputAddOne, vregInput, FP32_ONE, mask);
                    MicroAPI::Adds(vregInputMid, vregInputAddOne, FP32_NEG_ONE, mask);
                    MicroAPI::Div(vregInputMid, vregInput, vregInputMid, mask);
                    MicroAPI::Log(vregOutput, vregInputAddOne, mask);
                    MicroAPI::Mul(vregOutput, vregOutput, vregInputMid, mask);
                    MicroAPI::CompareScalar<float, CMPMODE::NE>(cmpMaskReg, vregInputAddOne, FP32_ONE, mask);
                    MicroAPI::Select(vregOutput, vregOutput, vregInput, cmpMaskReg);
                    MicroAPI::CompareScalar<float, CMPMODE::NE>(cmpMaskReg, vregInputAddOne, FP32_INF, mask);
                    MicroAPI::Duplicate(vregInputMid, FP32_INF, mask);
                    MicroAPI::Select(vregOutput, vregOutput, vregInputMid, cmpMaskReg);

                    MicroAPI::Cast<T, float, castTrait1>(vregOutput16, vregOutput, mask);
                    // OpCopyOut
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput16, mask);
                }
            }
        }
#endif
    }
};
} // namespace Log1pDag1

template <typename U>
struct Log1pDAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpResult1 = Bind<Log1pDag1::Log1pCustom<U>, OpCopyIn0>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResult1>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace Log1pOp
#endif // CANN_CUSTOM_OPS_LOG1P_DAG_H