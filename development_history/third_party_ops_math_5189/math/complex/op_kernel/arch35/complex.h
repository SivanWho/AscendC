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
 * \file complex.h
 * \brief Complex operator SIMT kernel
 */
#ifndef COMPLEX_SIMT_H
#define COMPLEX_SIMT_H

#include "kernel_operator.h"
#include "complex_struct.h"
#ifdef __CCE_AICORE__
#include "simt_api/asc_simt.h"
#endif

namespace ComplexOp {
using namespace AscendC;

constexpr int32_t SIMT_LAUNCH_BOUND_MAX = 2048;

// Internal stride parameter struct (kernel side uses uint64_t)
struct ComplexStridePara {
    uint64_t ms[COMPLEX_MAX_DIM];
    uint64_t realS[COMPLEX_MAX_DIM];
    uint64_t imagS[COMPLEX_MAX_DIM];
};

// ---------------------------------------------------------------------------
// StoreComplex
//   将 (re, im) 打包成单个等宽整型并一次性写出 outGm[v],强制单条宽 store:
//     half  -> uint32_t (4B):  [imag:16 | real:16]
//     float -> uint64_t (8B):  [imag:32 | real:32]
// ---------------------------------------------------------------------------
__simt_callee__ inline void StoreComplex(
    __gm__ DTYPE_OUT* outGm, uint64_t v, DTYPE_REAL re, DTYPE_REAL im)
{
    if constexpr (sizeof(DTYPE_REAL) == 2) {
        uint32_t lo = *reinterpret_cast<uint16_t*>(&re);
        uint32_t hi = *reinterpret_cast<uint16_t*>(&im);
        reinterpret_cast<__gm__ uint32_t*>(outGm)[v] = lo | (hi << 16);
    } else {
        uint64_t lo = *reinterpret_cast<uint32_t*>(&re);
        uint64_t hi = *reinterpret_cast<uint32_t*>(&im);
        reinterpret_cast<__gm__ uint64_t*>(outGm)[v] = lo | (hi << 32);
    }
}

class ComplexSimt {
public:
    __aicore__ inline ComplexSimt() {}
    __aicore__ inline ~ComplexSimt() {}

    __aicore__ inline void Init(GM_ADDR real, GM_ADDR imag, GM_ADDR out,
                                 const ComplexTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __simt_vf__ LAUNCH_BOUND(SIMT_LAUNCH_BOUND_MAX)
    static void SimtComplexCompute(
        __gm__ DTYPE_REAL* realGm,
        __gm__ DTYPE_REAL* imagGm,
        __gm__ DTYPE_OUT* outGm,
        uint64_t startIdx,
        uint64_t count,
        int32_t dimNum,
        int32_t elementsPerThread,
        int32_t mode,
        ComplexStridePara para);

private:
    GlobalTensor<DTYPE_REAL> realGm_;
    GlobalTensor<DTYPE_REAL> imagGm_;
    GlobalTensor<DTYPE_OUT> outGm_;

    uint64_t totalElements_{0};
    uint64_t gridDim_{0};
    uint64_t blockDim_{0};
    uint64_t elementsPerThread_{1};
    uint64_t elementsPerBlock_{0};
    uint64_t formerBlock_{0};

    int32_t dimNum_{0};
    int32_t mode_{MODE_GENERAL_BROADCAST};

    ComplexStridePara para_;
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
__aicore__ inline void ComplexSimt::Init(
    GM_ADDR real, GM_ADDR imag, GM_ADDR out,
    const ComplexTilingData& tilingData)
{
    totalElements_     = static_cast<uint64_t>(tilingData.totalElements);
    gridDim_           = static_cast<uint64_t>(tilingData.gridDim);
    blockDim_          = static_cast<uint64_t>(tilingData.blockDim);
    elementsPerThread_ = static_cast<uint64_t>(tilingData.elementsPerThread);
    elementsPerBlock_  = static_cast<uint64_t>(tilingData.elementsPerBlock);
    formerBlock_       = static_cast<uint64_t>(tilingData.formerBlock);
    dimNum_            = static_cast<int32_t>(tilingData.dimNum);
    mode_             = tilingData.mode;

    for (int i = 0; i < COMPLEX_MAX_DIM; i++) {
        para_.ms[i]    = tilingData.mergedStride[i];
        para_.realS[i] = tilingData.realStride[i];
        para_.imagS[i] = tilingData.imagStride[i];
    }

    realGm_.SetGlobalBuffer((__gm__ DTYPE_REAL*)real);
    imagGm_.SetGlobalBuffer((__gm__ DTYPE_REAL*)imag);
    outGm_.SetGlobalBuffer((__gm__ DTYPE_OUT*)out);
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------
__aicore__ inline void ComplexSimt::Process()
{
    uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());

    uint64_t startIdx = 0;
    uint64_t count = 0;

    if (blockIdx < formerBlock_) {
        count = elementsPerBlock_ + 1;
        startIdx = blockIdx * count;
    } else {
        count = elementsPerBlock_;
        startIdx = formerBlock_ * (elementsPerBlock_ + 1)
                 + (blockIdx - formerBlock_) * elementsPerBlock_;
    }

    if (count == 0) {
        return;
    }

    asc_vf_call<SimtComplexCompute>(
        dim3(static_cast<uint32_t>(blockDim_)),
        (__gm__ DTYPE_REAL*)(realGm_.GetPhyAddr()),
        (__gm__ DTYPE_REAL*)(imagGm_.GetPhyAddr()),
        (__gm__ DTYPE_OUT*)(outGm_.GetPhyAddr()),
        startIdx,
        count,
        dimNum_,
        static_cast<int32_t>(elementsPerThread_),
        mode_,
        para_
    );
}

// ---------------------------------------------------------------------------
// SimtComplexCompute — SIMT VF function
// ---------------------------------------------------------------------------
__simt_vf__ LAUNCH_BOUND(SIMT_LAUNCH_BOUND_MAX)
/* static */ void ComplexSimt::SimtComplexCompute(
    __gm__ DTYPE_REAL* realGm,
    __gm__ DTYPE_REAL* imagGm,
    __gm__ DTYPE_OUT* outGm,
    uint64_t startIdx,
    uint64_t count,
    int32_t dimNum,
    int32_t elementsPerThread,
    int32_t mode,
    ComplexStridePara para)
{
    const uint64_t tid = static_cast<uint64_t>(threadIdx.x);
    // blockDim.x is uint32 under SIMT, safe to cast
    const uint64_t B = static_cast<uint64_t>(blockDim.x);
    const uint64_t E = static_cast<uint64_t>(elementsPerThread);
    const int32_t iDimNum = dimNum;

    if (mode == MODE_FAST_CONTIGUOUS) {
        // ===================================================================
        // Fast Path: no broadcast, all tensors contiguous
        //   一次宽 store 写出 { realGm[v], imagGm[v] }
        // ===================================================================
        for (uint64_t base = tid; base < count; base += B * E) {  // 这里只是兜底性设计，实际当前B * E >= count，外层循环只有1次
            for (uint64_t e = 0; e < E; ++e) {
                uint64_t local = base + e * B;
                if (local >= count) {
                    break;
                }
                uint64_t v = startIdx + local;
                StoreComplex(outGm, v, realGm[v], imagGm[v]);
            }
        }
    } else {
        // ===================================================================
        // General Broadcast Path: stride-based broadcast addressing
        //   stride=0 handles broadcast without branching
        // ===================================================================
        for (uint64_t base = tid; base < count; base += B * E) {
            for (uint64_t e = 0; e < E; ++e) {
                uint64_t local = base + e * B;
                if (local >= count) {
                    continue;
                }
                uint64_t v = startIdx + local;

                // Stride-based address calculation
                uint64_t realBase = 0;
                uint64_t imagBase = 0;
                uint64_t vv = v;
                for (int32_t d = 0; d < iDimNum; ++d) {
                    uint64_t coord = vv / para.ms[d];
                    vv -= coord * para.ms[d];
                    realBase += coord * para.realS[d];
                    imagBase += coord * para.imagS[d];
                }

                StoreComplex(outGm, v, realGm[realBase], imagGm[imagBase]);
            }
        }
    }
}

}  // namespace ComplexOp

#endif  // COMPLEX_SIMT_H