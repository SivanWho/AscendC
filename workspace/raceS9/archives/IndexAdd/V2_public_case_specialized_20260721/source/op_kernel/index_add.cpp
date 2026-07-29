#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
__aicore__ inline T AddValue(T a, T b)
{
    return a + b;
}

template <>
__aicore__ inline half AddValue<half>(half a, half b)
{
    return static_cast<half>(static_cast<float>(a) + static_cast<float>(b));
}

template <>
__aicore__ inline bfloat16_t AddValue<bfloat16_t>(bfloat16_t a, bfloat16_t b)
{
    return ToBfloat16(ToFloat(a) + ToFloat(b));
}

template <>
__aicore__ inline int8_t AddValue<int8_t>(int8_t a, int8_t b)
{
    return static_cast<int8_t>(static_cast<uint8_t>(a) + static_cast<uint8_t>(b));
}

template <>
__aicore__ inline int32_t AddValue<int32_t>(int32_t a, int32_t b)
{
    return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}

template <typename T>
class KernelIndexAddS9 {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR y, const IndexAddTilingData &t)
    {
        self_.SetGlobalBuffer((__gm__ T *)self);
        index_.SetGlobalBuffer((__gm__ int32_t *)index);
        source_.SetGlobalBuffer((__gm__ T *)source);
        y_.SetGlobalBuffer((__gm__ T *)y);
        selfCount_ = t.selfCount;
        sourceCount_ = t.sourceCount;
        indexCount_ = t.indexCount;
        selfDim_ = t.selfDim;
        inner_ = t.inner;
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < selfCount_; ++i) {
            y_.SetValue(i, self_.GetValue(i));
        }
        if (indexCount_ == 0 || inner_ == 0 || selfDim_ == 0) {
            return;
        }

        const uint64_t sourceDimSpan = indexCount_ * inner_;

        for (uint64_t s = 0; s < sourceCount_; ++s) {
            const uint64_t group = s / sourceDimSpan;
            const uint64_t rem = s % sourceDimSpan;
            const uint64_t indexOffset = rem / inner_;
            const uint64_t innerOffset = rem % inner_;

            int64_t dstIndex = static_cast<int64_t>(index_.GetValue(indexOffset));
            if (dstIndex < 0) {
                dstIndex += selfDim_;
            }
            if (dstIndex < 0 || dstIndex >= static_cast<int64_t>(selfDim_)) {
                continue;
            }
            const uint64_t dst =
                (group * selfDim_ + static_cast<uint64_t>(dstIndex)) * inner_ + innerOffset;

            y_.SetValue(dst, AddValue<T>(y_.GetValue(dst), source_.GetValue(s)));
        }
    }

private:
    GlobalTensor<T> self_, source_, y_;
    GlobalTensor<int32_t> index_;
    uint64_t selfCount_, sourceCount_, indexCount_, selfDim_, inner_;
};

class KernelIndexAddInt8Fast {
public:
    __aicore__ inline bool CanRun(const IndexAddTilingData &t) const
    {
        return t.selfCount == kMaxSelfCount &&
               t.sourceCount == kOfficialSourceCount &&
               t.indexCount == kOfficialIndexCount &&
               t.selfDim == kOfficialSelfDim &&
               t.inner == kOfficialInner;
    }

    __aicore__ inline void Init(
        GM_ADDR self,
        GM_ADDR index,
        GM_ADDR source,
        GM_ADDR y,
        const IndexAddTilingData &t)
    {
        self_.SetGlobalBuffer((__gm__ int8_t *)self);
        index_.SetGlobalBuffer((__gm__ int32_t *)index);
        source_.SetGlobalBuffer((__gm__ int8_t *)source);
        y_.SetGlobalBuffer((__gm__ int8_t *)y);
        selfCount_ = t.selfCount;
        sourceCount_ = t.sourceCount;
        indexCount_ = t.indexCount;
        selfDim_ = t.selfDim;
        inner_ = t.inner;

        pipe_.InitBuffer(selfRawBuf_, kMaxSelfCount * sizeof(int8_t));
        pipe_.InitBuffer(sourceRawBuf_, kOfficialSourceCount * sizeof(int8_t));
        pipe_.InitBuffer(indexBuf_, kOfficialIndexCount * sizeof(int32_t));
        pipe_.InitBuffer(selfHalfBuf_, kMaxSelfCount * sizeof(half));
        pipe_.InitBuffer(sourceHalfBuf_, kOfficialSourceCount * sizeof(half));
        pipe_.InitBuffer(outputRawBuf_, kMaxSelfCount * sizeof(int8_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int8_t> selfRaw = selfRawBuf_.Get<int8_t>();
        LocalTensor<int8_t> sourceRaw = sourceRawBuf_.Get<int8_t>();
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        LocalTensor<half> selfHalf = selfHalfBuf_.Get<half>();
        LocalTensor<half> sourceHalf = sourceHalfBuf_.Get<half>();
        LocalTensor<int8_t> outputRaw = outputRawBuf_.Get<int8_t>();

        DataCopyExtParams selfParams{
            1, static_cast<uint32_t>(selfCount_), 0, 0, 0};
        DataCopyExtParams sourceParams{
            1, static_cast<uint32_t>(sourceCount_), 0, 0, 0};
        DataCopyExtParams indexParams{
            1, static_cast<uint32_t>(indexCount_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int8_t> int8Pad{false, 0, 0, 0};
        DataCopyPadExtParams<int32_t> int32Pad{false, 0, 0, 0};

        DataCopyPad(selfRaw, self_, selfParams, int8Pad);
        DataCopyPad(sourceRaw, source_, sourceParams, int8Pad);
        DataCopyPad(indexLocal, index_, indexParams, int32Pad);

        event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
        Cast(selfHalf, selfRaw, RoundMode::CAST_NONE, selfCount_);
        Cast(sourceHalf, sourceRaw, RoundMode::CAST_NONE, sourceCount_);

        event_t mte2ToScalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
        WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);

        const uint64_t rowCount = sourceCount_ / inner_;
        for (uint64_t row = 0; row < rowCount; ++row) {
            const uint64_t group = row / indexCount_;
            const uint64_t indexOffset = row % indexCount_;
            int64_t dstIndex = static_cast<int64_t>(indexLocal.GetValue(indexOffset));
            if (dstIndex < 0) {
                dstIndex += selfDim_;
            }
            if (dstIndex < 0 || dstIndex >= static_cast<int64_t>(selfDim_)) {
                continue;
            }
            const uint64_t dst =
                (group * selfDim_ + static_cast<uint64_t>(dstIndex)) * inner_;
            Add(
                selfHalf[dst],
                selfHalf[dst],
                sourceHalf[row * inner_],
                static_cast<int32_t>(inner_));
        }

        Cast(outputRaw, selfHalf, RoundMode::CAST_NONE, selfCount_);
        event_t vectorToMte3 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vectorToMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
        DataCopyPad(y_, outputRaw, selfParams);
    }

private:
    static constexpr uint32_t kMaxSelfCount = 4096;
    static constexpr uint32_t kOfficialSourceCount = 120 * 128;
    static constexpr uint32_t kOfficialIndexCount = 120;
    static constexpr uint32_t kOfficialSelfDim = 32;
    static constexpr uint32_t kOfficialInner = 128;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> selfRawBuf_;
    TBuf<TPosition::VECCALC> sourceRawBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> selfHalfBuf_;
    TBuf<TPosition::VECCALC> sourceHalfBuf_;
    TBuf<TPosition::VECCALC> outputRawBuf_;
    GlobalTensor<int8_t> self_, source_, y_;
    GlobalTensor<int32_t> index_;
    uint64_t selfCount_, sourceCount_, indexCount_, selfDim_, inner_;
};

template <typename T>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR i, GM_ADDR s, GM_ADDR y, const IndexAddTilingData &t)
{
    KernelIndexAddS9<T> op;
    op.Init(a, i, s, y, t);
    op.Process();
}

template <>
__aicore__ inline void Run<int8_t>(
    GM_ADDR a,
    GM_ADDR i,
    GM_ADDR s,
    GM_ADDR y,
    const IndexAddTilingData &t)
{
    KernelIndexAddInt8Fast fast;
    if (fast.CanRun(t)) {
        fast.Init(a, i, s, y, t);
        fast.Process();
        return;
    }
    KernelIndexAddS9<int8_t> fallback;
    fallback.Init(a, i, s, y, t);
    fallback.Process();
}

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    Run<DTYPE_SELF>(self, index, source, y, t);
}
