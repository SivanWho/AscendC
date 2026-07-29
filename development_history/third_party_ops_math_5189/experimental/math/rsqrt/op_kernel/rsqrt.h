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
 * \file rsqrt.h
 * \brief
 */
#ifndef __RSQRT_H__
#define __RSQRT_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "rsqrt_tiling_data.h"
#include "rsqrt_tiling_key.h"

namespace NsRsqrt {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelRsqrt {
    using T = TYPE_X;

public:
    __aicore__ inline KernelRsqrt() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
        uint32_t finalSmallTileNum, uint32_t tileDataNum, uint32_t smallTailDataNum, uint32_t bigTailDataNum,
        uint32_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t blockIdx = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        if constexpr (!std::is_same_v<T, float32_t>) {
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;

        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;

        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);

        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        if constexpr (!std::is_same_v<T, float32_t>) {
            LocalTensor<float> p1 = tmp1.Get<float>();
            LocalTensor<float> p2 = tmp2.Get<float>();
            Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Duplicate<float>(p2, 1.0f, this->processDataNum);
            Sqrt(p1, p1, this->processDataNum);
            Div(p1, p2, p1, this->processDataNum);
            Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);
        } else {
            Duplicate<float>(yLocal, 1.0f, this->processDataNum);
            Sqrt(xLocal, xLocal, this->processDataNum);
            Div(yLocal, yLocal, xLocal, this->processDataNum);
        }
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_X> yGm;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
};

} // namespace NsRsqrt

#endif // RSQRT_H
