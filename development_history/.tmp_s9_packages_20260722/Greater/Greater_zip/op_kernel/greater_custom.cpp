#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
__aicore__ inline bool IsGreater(T a, T b)
{
    return a > b;
}

union HalfBits {
    half value;
    uint16_t bits;
};

union Bfloat16Bits {
    bfloat16_t value;
    uint16_t bits;
};

__aicore__ inline bool IsGreaterIeee16(
    uint16_t a,
    uint16_t b,
    uint16_t exponentMask,
    uint16_t mantissaMask)
{
    const bool aNan = (a & exponentMask) == exponentMask && (a & mantissaMask) != 0;
    const bool bNan = (b & exponentMask) == exponentMask && (b & mantissaMask) != 0;
    if (aNan || bNan) {
        return false;
    }
    if ((a & 0x7FFFU) == 0 && (b & 0x7FFFU) == 0) {
        return false;
    }
    const bool aNegative = (a & 0x8000U) != 0;
    const bool bNegative = (b & 0x8000U) != 0;
    if (aNegative != bNegative) {
        return !aNegative;
    }
    return aNegative ? a < b : a > b;
}

template <>
__aicore__ inline bool IsGreater<half>(half a, half b)
{
    HalfBits aBits;
    HalfBits bBits;
    aBits.value = a;
    bBits.value = b;
    return IsGreaterIeee16(aBits.bits, bBits.bits, 0x7C00U, 0x03FFU);
}

template <>
__aicore__ inline bool IsGreater<bfloat16_t>(bfloat16_t a, bfloat16_t b)
{
    Bfloat16Bits aBits;
    Bfloat16Bits bBits;
    aBits.value = a;
    bBits.value = b;
    return IsGreaterIeee16(aBits.bits, bBits.bits, 0x7F80U, 0x007FU);
}

template <typename T>
class KernelGreaterS9 {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterCustomTilingData &t)
    {
        a_.SetGlobalBuffer((__gm__ T *)a);
        b_.SetGlobalBuffer((__gm__ T *)b);
        y_.SetGlobalBuffer((__gm__ int8_t *)y);
        rank_ = t.rank;
        blocks_ = t.blockDim;
        total_ = t.outputCount;
        for (uint32_t i = 0; i < rank_; ++i) {
            shape_[i] = t.outputShape[i];
            sa_[i] = t.x1Strides[i];
            sb_[i] = t.x2Strides[i];
        }
    }

    __aicore__ inline void Process()
    {
        for (uint64_t idx = GetBlockIdx(); idx < total_; idx += blocks_) {
            uint64_t rem = idx, oa = 0, ob = 0;
            for (int32_t d = rank_ - 1; d >= 0; --d) {
                const uint64_t c = rem % shape_[d];
                rem /= shape_[d];
                oa += c * sa_[d];
                ob += c * sb_[d];
            }
            y_.SetValue(idx, IsGreater<T>(a_.GetValue(oa), b_.GetValue(ob)) ? 1 : 0);
        }
    }

private:
    GlobalTensor<T> a_, b_;
    GlobalTensor<int8_t> y_;
    uint64_t shape_[8], sa_[8], sb_[8], total_;
    uint32_t rank_, blocks_;
};

template <typename T>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterCustomTilingData &t)
{
    KernelGreaterS9<T> op;
    op.Init(a, b, y, t);
    op.Process();
}

class KernelGreaterFp16Fast {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterCustomTilingData &t)
    {
        const uint64_t perCore = t.outputCount / t.blockDim;
        const uint64_t offset = perCore * GetBlockIdx();
        count_ = static_cast<uint32_t>(perCore);
        tile_ = count_ < 4096 ? count_ : 4096;
        a_.SetGlobalBuffer((__gm__ half *)a + offset, count_);
        b_.SetGlobalBuffer((__gm__ half *)b + offset, count_);
        y_.SetGlobalBuffer((__gm__ int8_t *)y + offset, count_);
        pipe_.InitBuffer(aQueue_, 1, tile_ * sizeof(half));
        pipe_.InitBuffer(bQueue_, 1, tile_ * sizeof(half));
        pipe_.InitBuffer(yQueue_, 1, tile_ * sizeof(int8_t));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < count_; offset += tile_) {
            const uint32_t current = count_ - offset < tile_ ? count_ - offset : tile_;
            LocalTensor<half> aLocal = aQueue_.AllocTensor<half>();
            LocalTensor<half> bLocal = bQueue_.AllocTensor<half>();
            DataCopy(aLocal, a_[offset], current);
            DataCopy(bLocal, b_[offset], current);
            aQueue_.EnQue(aLocal);
            bQueue_.EnQue(bLocal);

            aLocal = aQueue_.DeQue<half>();
            bLocal = bQueue_.DeQue<half>();
            LocalTensor<int8_t> yLocal = yQueue_.AllocTensor<int8_t>();
            Compare(yLocal, aLocal, bLocal, CMPMODE::GT, current);
            Duplicate(aLocal, static_cast<half>(1), current);
            Select(aLocal, yLocal, aLocal, static_cast<half>(0),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, current);
            Cast(yLocal, aLocal, RoundMode::CAST_NONE, current);
            aQueue_.FreeTensor(aLocal);
            bQueue_.FreeTensor(bLocal);
            yQueue_.EnQue(yLocal);

            yLocal = yQueue_.DeQue<int8_t>();
            DataCopy(y_[offset], yLocal, current);
            yQueue_.FreeTensor(yLocal);
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> aQueue_, bQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    GlobalTensor<half> a_, b_;
    GlobalTensor<int8_t> y_;
    uint32_t count_, tile_;
};

extern "C" __global__ __aicore__ void greater_custom(
    GM_ADDR a,
    GM_ADDR b,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    if (t.fastPath == 1) {
        KernelGreaterFp16Fast op;
        op.Init(a, b, y, t);
        op.Process();
        return;
    }
    if (t.dtypeCode == 0) {
        Run<half>(a, b, y, t);
    } else if (t.dtypeCode == 1) {
        Run<float>(a, b, y, t);
    } else if (t.dtypeCode == 2) {
        Run<int32_t>(a, b, y, t);
    } else if (t.dtypeCode == 3) {
        Run<int8_t>(a, b, y, t);
    } else {
        Run<bfloat16_t>(a, b, y, t);
    }
}
