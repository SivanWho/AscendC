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
 * \file ceil_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_CEIL_DAG_H
#define CANN_CUSTOM_OPS_CEIL_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace CeilOp {
using namespace Ops::Base;
const uint32_t UINT32_SIGN = 0x80000000;
const uint16_t UINT16_SIGN = 0x8000;

namespace CeilDag1 {
template <class T>
struct CeilCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline CeilCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
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

                    MicroAPI::Truncate<T, RoundMode::CAST_CEIL, MicroAPI::MaskMergeMode::ZEROING>(
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

                    MicroAPI::Truncate<T, RoundMode::CAST_CEIL, MicroAPI::MaskMergeMode::ZEROING>(
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
} // namespace CeilDag1

template <typename U>
struct CeilDAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpLogResult = Bind<CeilDag1::CeilCustom<U>, OpCopyIn0>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpLogResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace CeilOp
#endif // CANN_CUSTOM_OPS_CEIL_DAG_H