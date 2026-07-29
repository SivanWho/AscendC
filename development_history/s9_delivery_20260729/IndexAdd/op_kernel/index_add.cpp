#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTileElements = 8192;

template <typename T>
__aicore__ inline void CopyIn(
    const LocalTensor<T> &destination,
    const GlobalTensor<T> &source,
    uint32_t count)
{
    DataCopyExtParams params{
        1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(destination, source, params, pad);
}

template <typename T>
__aicore__ inline void CopyOut(
    const GlobalTensor<T> &destination,
    const LocalTensor<T> &source,
    uint32_t count)
{
    DataCopyExtParams params{
        1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(destination, source, params);
}

template <typename T>
__aicore__ inline T AddValue(T left, T right)
{
    return left + right;
}

template <>
__aicore__ inline half AddValue<half>(half left, half right)
{
    return static_cast<half>(
        static_cast<float>(left) + static_cast<float>(right));
}

template <>
__aicore__ inline bfloat16_t AddValue<bfloat16_t>(
    bfloat16_t left,
    bfloat16_t right)
{
    return static_cast<bfloat16_t>(
        static_cast<float>(left) + static_cast<float>(right));
}

template <>
__aicore__ inline int8_t AddValue<int8_t>(int8_t left, int8_t right)
{
    return static_cast<int8_t>(
        static_cast<uint8_t>(left) + static_cast<uint8_t>(right));
}

template <>
__aicore__ inline int32_t AddValue<int32_t>(int32_t left, int32_t right)
{
    return static_cast<int32_t>(
        static_cast<uint32_t>(left) + static_cast<uint32_t>(right));
}
}  // namespace

template <typename T>
class KernelIndexAddSafe {
public:
    __aicore__ inline void Init(
        GM_ADDR self,
        GM_ADDR index,
        GM_ADDR source,
        GM_ADDR y,
        const IndexAddTilingData &tiling)
    {
        self_.SetGlobalBuffer((__gm__ T *)self);
        index_.SetGlobalBuffer((__gm__ int32_t *)index);
        source_.SetGlobalBuffer((__gm__ T *)source);
        y_.SetGlobalBuffer((__gm__ T *)y);
        selfCount_ = tiling.selfCount;
        sourceCount_ = tiling.sourceCount;
        indexCount_ = tiling.indexCount;
        selfDim_ = tiling.selfDim;
        inner_ = tiling.inner;
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < selfCount_; ++i) {
            y_.SetValue(i, self_.GetValue(i));
        }

        if (indexCount_ == 0 || inner_ == 0 || selfDim_ == 0) {
            return;
        }

        for (uint64_t sourceOffset = 0; sourceOffset < sourceCount_;
             ++sourceOffset) {
            const uint64_t group =
                sourceOffset / (indexCount_ * inner_);
            const uint64_t remainder =
                sourceOffset % (indexCount_ * inner_);
            const uint64_t indexOffset = remainder / inner_;
            const uint64_t innerOffset = remainder % inner_;

            int64_t destinationIndex = index_.GetValue(indexOffset);
            if (destinationIndex < 0) {
                destinationIndex += static_cast<int64_t>(selfDim_);
            }
            if (destinationIndex < 0 ||
                destinationIndex >= static_cast<int64_t>(selfDim_)) {
                continue;
            }

            const uint64_t outputOffset =
                (group * selfDim_ +
                 static_cast<uint64_t>(destinationIndex)) *
                    inner_ +
                innerOffset;
            y_.SetValue(
                outputOffset,
                AddValue<T>(
                    y_.GetValue(outputOffset),
                    source_.GetValue(sourceOffset)));
        }
    }

private:
    GlobalTensor<T> self_;
    GlobalTensor<T> source_;
    GlobalTensor<T> y_;
    GlobalTensor<int32_t> index_;
    uint64_t selfCount_;
    uint64_t sourceCount_;
    uint64_t indexCount_;
    uint64_t selfDim_;
    uint64_t inner_;
};

class KernelIndexAddBfloat16 {
public:
    __aicore__ inline void Init(
        GM_ADDR self,
        GM_ADDR index,
        GM_ADDR source,
        GM_ADDR y,
        const IndexAddTilingData &tiling)
    {
        self_.SetGlobalBuffer((__gm__ bfloat16_t *)self);
        index_.SetGlobalBuffer((__gm__ int32_t *)index);
        source_.SetGlobalBuffer((__gm__ bfloat16_t *)source);
        y_.SetGlobalBuffer((__gm__ bfloat16_t *)y);
        selfCount_ = tiling.selfCount;
        indexCount_ = tiling.indexCount;
        selfDim_ = tiling.selfDim;
        inner_ = tiling.inner;
        const uint64_t denominator = selfDim_ * inner_;
        outer_ = denominator == 0 ? 0 : selfCount_ / denominator;
        chunksPerRow_ =
            inner_ == 0 ? 0 : (inner_ + kTileElements - 1) / kTileElements;
        taskCount_ = outer_ * selfDim_ * chunksPerRow_;

        pipe_.InitBuffer(accRawBuf_, kTileElements * sizeof(bfloat16_t));
        pipe_.InitBuffer(sourceRawBuf_, kTileElements * sizeof(bfloat16_t));
        pipe_.InitBuffer(accFloatBuf_, kTileElements * sizeof(float));
        pipe_.InitBuffer(sourceFloatBuf_, kTileElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<bfloat16_t> accRaw = accRawBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> sourceRaw =
            sourceRawBuf_.Get<bfloat16_t>();
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> sourceFloat = sourceFloatBuf_.Get<float>();

        for (uint64_t task = 0; task < taskCount_; ++task) {
            const uint64_t rowTask = task / chunksPerRow_;
            const uint64_t chunk = task % chunksPerRow_;
            const uint64_t group = rowTask / selfDim_;
            const uint64_t row = rowTask % selfDim_;
            const uint64_t column = chunk * kTileElements;
            const uint32_t length = static_cast<uint32_t>(
                inner_ - column < kTileElements
                    ? inner_ - column
                    : kTileElements);
            const uint64_t outputOffset =
                (group * selfDim_ + row) * inner_ + column;

            CopyIn(accRaw, self_[outputOffset], length);
            event_t selfReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(selfReady);
            WaitFlag<HardEvent::MTE2_V>(selfReady);
            Cast(accFloat, accRaw, RoundMode::CAST_NONE, length);

            for (uint64_t i = 0; i < indexCount_; ++i) {
                int64_t destination = index_.GetValue(i);
                if (destination < 0) {
                    destination += static_cast<int64_t>(selfDim_);
                }
                if (destination != static_cast<int64_t>(row)) {
                    continue;
                }
                const uint64_t sourceOffset =
                    (group * indexCount_ + i) * inner_ + column;
                CopyIn(sourceRaw, source_[sourceOffset], length);
                event_t sourceReady = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(sourceReady);
                WaitFlag<HardEvent::MTE2_V>(sourceReady);
                Cast(sourceFloat, sourceRaw, RoundMode::CAST_NONE, length);
                Add(accFloat, accFloat, sourceFloat, length);
                Cast(accRaw, accFloat, RoundMode::CAST_RINT, length);
                Cast(accFloat, accRaw, RoundMode::CAST_NONE, length);

                event_t sourceConsumed = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(sourceConsumed);
                WaitFlag<HardEvent::V_MTE2>(sourceConsumed);
            }

            Cast(accRaw, accFloat, RoundMode::CAST_RINT, length);
            event_t outputReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(outputReady);
            WaitFlag<HardEvent::V_MTE3>(outputReady);
            CopyOut(y_[outputOffset], accRaw, length);
            event_t outputConsumed = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(outputConsumed);
            WaitFlag<HardEvent::MTE3_MTE2>(outputConsumed);
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> accRawBuf_;
    TBuf<TPosition::VECCALC> sourceRawBuf_;
    TBuf<TPosition::VECCALC> accFloatBuf_;
    TBuf<TPosition::VECCALC> sourceFloatBuf_;
    GlobalTensor<bfloat16_t> self_;
    GlobalTensor<bfloat16_t> source_;
    GlobalTensor<bfloat16_t> y_;
    GlobalTensor<int32_t> index_;
    uint64_t selfCount_;
    uint64_t indexCount_;
    uint64_t selfDim_;
    uint64_t inner_;
    uint64_t outer_;
    uint64_t chunksPerRow_;
    uint64_t taskCount_;
};

template <typename T>
__aicore__ inline void Run(
    GM_ADDR self,
    GM_ADDR index,
    GM_ADDR source,
    GM_ADDR y,
    const IndexAddTilingData &tiling)
{
    KernelIndexAddSafe<T> op;
    op.Init(self, index, source, y, tiling);
    op.Process();
}

template <>
__aicore__ inline void Run<bfloat16_t>(
    GM_ADDR self,
    GM_ADDR index,
    GM_ADDR source,
    GM_ADDR y,
    const IndexAddTilingData &tiling)
{
    KernelIndexAddBfloat16 op;
    op.Init(self, index, source, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR self,
    GM_ADDR index,
    GM_ADDR source,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    Run<DTYPE_SELF>(self, index, source, y, tilingData);
}
