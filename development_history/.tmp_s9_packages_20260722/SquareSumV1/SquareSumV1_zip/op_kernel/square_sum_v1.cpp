#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
__aicore__ inline float ToFloat(T v)
{
    return static_cast<float>(v);
}

template <>
__aicore__ inline float ToFloat<float>(float v)
{
    return v;
}

union Bfloat16Bits {
    bfloat16_t value;
    uint16_t bits;
};

union FloatBits {
    float value;
    uint32_t bits;
};

template <>
__aicore__ inline float ToFloat<bfloat16_t>(bfloat16_t v)
{
    Bfloat16Bits src;
    src.value = v;
    FloatBits dst;
    dst.bits = static_cast<uint32_t>(src.bits) << 16;
    return dst.value;
}

template <typename T>
__aicore__ inline T FromFloat(float v)
{
    return static_cast<T>(v);
}

template <>
__aicore__ inline float FromFloat<float>(float v)
{
    return v;
}

template <>
__aicore__ inline bfloat16_t FromFloat<bfloat16_t>(float v)
{
    FloatBits src;
    src.value = v;
    const uint32_t roundingBias = 0x7FFFU + ((src.bits >> 16) & 1U);
    src.bits += roundingBias;
    Bfloat16Bits dst;
    dst.bits = static_cast<uint16_t>(src.bits >> 16);
    return dst.value;
}

template <typename T>
class KernelSquareSumV1 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData &t)
    {
        x_.SetGlobalBuffer((__gm__ T *)x);
        y_.SetGlobalBuffer((__gm__ T *)y);
        rank_ = t.rank;
        blocks_ = t.blockDim;
        outCount_ = t.outputCount;
        reduceCount_ = t.reduceCount;
        for (uint32_t d = 0; d < rank_; ++d) {
            shape_[d] = t.inputShape[d];
            stride_[d] = t.inputStride[d];
            reduced_[d] = t.isReduced[d];
        }
    }

    __aicore__ inline void Process()
    {
        for (uint64_t out = GetBlockIdx(); out < outCount_; out += blocks_) {
            uint64_t rem = out, base = 0;
            for (int32_t d = rank_ - 1; d >= 0; --d) {
                if (!reduced_[d]) {
                    uint64_t c = rem % shape_[d];
                    rem /= shape_[d];
                    base += c * stride_[d];
                }
            }
            float sum = 0.0f;
            for (uint64_t r = 0; r < reduceCount_; ++r) {
                uint64_t rr = r, off = base;
                for (int32_t d = rank_ - 1; d >= 0; --d) {
                    if (reduced_[d]) {
                        uint64_t c = rr % shape_[d];
                        rr /= shape_[d];
                        off += c * stride_[d];
                    }
                }
                float v = ToFloat<T>(x_.GetValue(off));
                sum += v * v;
            }
            y_.SetValue(out, FromFloat<T>(sum));
        }
    }

private:
    GlobalTensor<T> x_, y_;
    uint64_t shape_[8], stride_[8], outCount_, reduceCount_;
    uint8_t reduced_[8];
    uint32_t rank_, blocks_;
};

template <typename T>
__aicore__ inline void Run(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData &t)
{
    KernelSquareSumV1<T> op;
    op.Init(x, y, t);
    op.Process();
}

class KernelSquareSumV1Fp16LastAxisFast {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SquareSumV1TilingData &t)
    {
        rows_ = static_cast<uint32_t>(t.outputCount);
        cols_ = static_cast<uint32_t>(t.inputShape[t.rank - 1]);
        paddedCols_ = (cols_ + 15) / 16 * 16;
        paddedTotal_ = rows_ * paddedCols_;
        const uint32_t outHalfBytes = (rows_ * sizeof(half) + 31) / 32 * 32;
        const uint32_t outFloatBytes = (rows_ * sizeof(float) + 31) / 32 * 32;
        x_.SetGlobalBuffer((__gm__ half *)x, rows_ * cols_);
        y_.SetGlobalBuffer((__gm__ half *)y, rows_);
        pipe_.InitBuffer(xQueue_, 1, paddedTotal_ * sizeof(half));
        pipe_.InitBuffer(yQueue_, 1, outHalfBytes);
        pipe_.InitBuffer(squareBuf_, paddedTotal_ * sizeof(float));
        pipe_.InitBuffer(sumBuf_, outFloatBytes);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> xLocal = xQueue_.AllocTensor<half>();
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rows_), static_cast<uint32_t>(cols_ * sizeof(half)), 0, 0, 0};
        DataCopyPadExtParams<half> padParams = {
            false, 0, static_cast<uint8_t>(paddedCols_ - cols_), static_cast<half>(0)};
        DataCopyPad(xLocal, x_, copyParams, padParams);
        xQueue_.EnQue(xLocal);

        xLocal = xQueue_.DeQue<half>();
        LocalTensor<float> squareLocal = squareBuf_.Get<float>();
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        LocalTensor<half> yLocal = yQueue_.AllocTensor<half>();
        Cast(squareLocal, xLocal, RoundMode::CAST_NONE, paddedTotal_);
        Mul(squareLocal, squareLocal, squareLocal, paddedTotal_);
        WholeReduceSum<float, true>(sumLocal, squareLocal, paddedCols_, rows_, 1, 1,
                                    paddedCols_ * sizeof(float) / 32);
        Cast(yLocal, sumLocal, RoundMode::CAST_NONE, rows_);
        xQueue_.FreeTensor(xLocal);
        yQueue_.EnQue(yLocal);

        yLocal = yQueue_.DeQue<half>();
        DataCopyExtParams outParams = {
            1, static_cast<uint32_t>(rows_ * sizeof(half)), 0, 0, 0};
        DataCopyPad(y_, yLocal, outParams);
        yQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<QuePosition::VECCALC> squareBuf_, sumBuf_;
    GlobalTensor<half> x_, y_;
    uint32_t rows_, cols_, paddedCols_, paddedTotal_;
};

extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    if (t.fastPath == 1) {
        KernelSquareSumV1Fp16LastAxisFast op;
        op.Init(x, y, t);
        op.Process();
        return;
    }
    if (t.dtypeCode == 0) {
        Run<half>(x, y, t);
    } else if (t.dtypeCode == 1) {
        Run<float>(x, y, t);
    } else {
        Run<bfloat16_t>(x, y, t);
    }
}
