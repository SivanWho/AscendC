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
 * \file complex_struct.h
 * \brief Complex operator shared data structures
 */
#ifndef COMPLEX_STRUCT_H_
#define COMPLEX_STRUCT_H_

#include <cstdint>

constexpr int64_t COMPLEX_MAX_DIM = 8;

// Execution modes
constexpr int32_t MODE_GENERAL_BROADCAST = 0;
constexpr int32_t MODE_FAST_CONTIGUOUS = 1;

#pragma pack(push, 8)
struct ComplexTilingData {
    int64_t totalElements;

    // SIMT scheduling parameters
    int64_t gridDim;
    int64_t blockDim;
    int64_t elementsPerThread;

    // Block-level element range allocation
    int64_t elementsPerBlock;
    int64_t formerBlock;

    // Shape / broadcast parameters
    int64_t dimNum;
    uint64_t mergedStride[COMPLEX_MAX_DIM];
    uint64_t realStride[COMPLEX_MAX_DIM];
    uint64_t imagStride[COMPLEX_MAX_DIM];

    // Execution mode: 0 = general broadcast, 1 = contiguous no-broadcast fast path
    int32_t mode;

    // dtype marker: 0 = float32->complex64, 1 = float16->complex32
    int32_t dtype;
};
#pragma pack(pop)

#endif  // COMPLEX_STRUCT_H_
