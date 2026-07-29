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
 * \file rint_dag.h
 * \brief rint_dag
 */

 #ifndef RINT_DAG_H
 #define RINT_DAG_H
 #include "atvoss/util/dag.h"
 #include "atvoss/util/vec.h"
 #include "atvoss/util/placeholder.h"

 namespace RintOp {
    using namespace AscendC;
    using namespace Ops::Base;

    constexpr uint32_t UINT32_SIGN = 0x80000000;
    constexpr uint16_t UINT16_SIGN = 0x8000;
    constexpr uint64_t VECTOR_REG_WIDTH = 256UL;

    template<class T>
    struct RintCalc : public Vec::ElemwiseUnaryOP<T, T> {
        __aicore__ inline RintCalc(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count) {
        #ifdef __CCE_AICORE__
            uint32_t dtypeSize = sizeof(T);
            uint32_t vlSize = VECTOR_REG_WIDTH / dtypeSize;
            uint16_t loopNum = (count + vlSize - 1) / vlSize;
            __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
            __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

            MicroAPI::RegTensor<T> inputReg;
            MicroAPI::RegTensor<T> outReg;
            MicroAPI::MaskReg mask;
            if constexpr (std::is_same_v<T, float>) {
                MicroAPI::RegTensor<uint32_t> resultReg;
                __VEC_SCOPE__
                {
                    for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                        mask = MicroAPI::UpdateMask<T>(count);
                        MicroAPI::LoadAlign(inputReg, srcAddr + loopIdx * vlSize);
                        MicroAPI::Truncate<T, RoundMode::CAST_RINT, MicroAPI::MaskMergeMode::ZEROING>(
                            outReg, inputReg, mask);
                        MicroAPI::Duplicate(resultReg, UINT32_SIGN, mask);
                        MicroAPI::And(resultReg, resultReg, (MicroAPI::RegTensor<uint32_t>&)inputReg, mask);
                        MicroAPI::Or(resultReg, resultReg, (MicroAPI::RegTensor<uint32_t>&)outReg, mask);
                        MicroAPI::StoreAlign(dstAddr + loopIdx * vlSize, (MicroAPI::RegTensor<T>&)resultReg, mask);
                    }
                }
            } else {
                MicroAPI::RegTensor<uint16_t> resultReg;
                __VEC_SCOPE__
                {
                    for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                        mask = MicroAPI::UpdateMask<T>(count);
                        MicroAPI::LoadAlign(inputReg, srcAddr + loopIdx * vlSize);
                        MicroAPI::Truncate<T, RoundMode::CAST_RINT, MicroAPI::MaskMergeMode::ZEROING>(
                            outReg, inputReg, mask);
                        MicroAPI::Duplicate(resultReg, UINT16_SIGN, mask);
                        MicroAPI::And(resultReg, resultReg, (MicroAPI::RegTensor<uint16_t>&)inputReg, mask);
                        MicroAPI::Or(resultReg, resultReg, (MicroAPI::RegTensor<uint16_t>&)outReg, mask);
                        MicroAPI::StoreAlign(dstAddr + loopIdx * vlSize, (MicroAPI::RegTensor<T>&)resultReg, mask);
                    }
                }
            }
        #endif
        }
    };
 
     template <typename T>
     struct RintDag {
         // 数据搬入
         using InputX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

         // 计算
         using OpResult = Bind<RintCalc<T>, InputX>;
 
         // Copy out
         using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;
 
         using Outputs = Elems<OpCopyOut>;
         using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
         using OpDag = DAGSch<Outputs, void, MemCfg>;
     };
 }
 
 #endif // RINT_DAG_H
