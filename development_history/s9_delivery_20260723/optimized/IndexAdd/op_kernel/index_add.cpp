#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTileElements = 8192;
constexpr uint32_t kMaxIndexCount = 8000;

template <typename T>
__aicore__ inline void CopyIn(
    const LocalTensor<T> &dst,
    const GlobalTensor<T> &src,
    uint32_t count)
{
    DataCopyExtParams params{
        1,
        static_cast<uint32_t>(count * sizeof(T)),
        0,
        0,
        0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(dst, src, params, pad);
}

template <typename T>
__aicore__ inline void CopyOut(
    const GlobalTensor<T> &dst,
    const LocalTensor<T> &src,
    uint32_t count)
{
    DataCopyExtParams params{
        1,
        static_cast<uint32_t>(count * sizeof(T)),
        0,
        0,
        0};
    DataCopyPad(dst, src, params);
}

struct TaskPosition {
    uint64_t group;
    uint64_t row;
    uint64_t column;
    uint32_t length;
};

__aicore__ inline TaskPosition DecodeTask(
    uint64_t task,
    uint64_t selfDim,
    uint64_t inner)
{
    const uint64_t chunksPerRow = (inner + kTileElements - 1) / kTileElements;
    const uint64_t rowTask = task / chunksPerRow;
    const uint64_t chunk = task % chunksPerRow;
    const uint64_t column = chunk * kTileElements;
    TaskPosition pos;
    pos.group = rowTask / selfDim;
    pos.row = rowTask % selfDim;
    pos.column = column;
    pos.length = static_cast<uint32_t>(
        inner - column < kTileElements ? inner - column : kTileElements);
    return pos;
}

__aicore__ inline int64_t NormalizeIndex(int32_t value, uint64_t selfDim)
{
    int64_t normalized = static_cast<int64_t>(value);
    if (normalized < 0) {
        normalized += static_cast<int64_t>(selfDim);
    }
    return normalized;
}
}  // namespace

template <typename T>
class KernelIndexAddVector {
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
        indexCount_ = tiling.indexCount;
        selfDim_ = tiling.selfDim;
        inner_ = tiling.inner;
        const uint64_t denominator = selfDim_ * inner_;
        outer_ = denominator == 0 ? 0 : selfCount_ / denominator;
        chunksPerRow_ =
            inner_ == 0 ? 0 : (inner_ + kTileElements - 1) / kTileElements;
        taskCount_ = outer_ * selfDim_ * chunksPerRow_;

        pipe_.InitBuffer(indexBuf_, kMaxIndexCount * sizeof(int32_t));
        pipe_.InitBuffer(accBuf_, kTileElements * sizeof(T));
        pipe_.InitBuffer(sourceBuf_, kTileElements * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        const bool indexFitsUb = indexCount_ <= kMaxIndexCount;
        if (indexFitsUb && indexCount_ > 0) {
            CopyIn(indexLocal, index_, static_cast<uint32_t>(indexCount_));
            event_t mte2ToScalar = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
        }

        LocalTensor<T> accLocal = accBuf_.Get<T>();
        LocalTensor<T> sourceLocal = sourceBuf_.Get<T>();
        const uint64_t blockCount = GetBlockNum();

        for (uint64_t task = GetBlockIdx(); task < taskCount_; task += blockCount) {
            const TaskPosition pos = DecodeTask(task, selfDim_, inner_);
            const uint64_t outputOffset =
                (pos.group * selfDim_ + pos.row) * inner_ + pos.column;

            CopyIn(accLocal, self_[outputOffset], pos.length);
            event_t selfReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(selfReady);
            WaitFlag<HardEvent::MTE2_V>(selfReady);

            for (uint64_t i = 0; i < indexCount_; ++i) {
                const int32_t rawIndex =
                    indexFitsUb ? indexLocal.GetValue(i) : index_.GetValue(i);
                const int64_t dstRow = NormalizeIndex(rawIndex, selfDim_);
                if (dstRow != static_cast<int64_t>(pos.row)) {
                    continue;
                }

                const uint64_t sourceOffset =
                    (pos.group * indexCount_ + i) * inner_ + pos.column;
                CopyIn(sourceLocal, source_[sourceOffset], pos.length);
                event_t sourceReady = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(sourceReady);
                WaitFlag<HardEvent::MTE2_V>(sourceReady);

                Add(accLocal, accLocal, sourceLocal, static_cast<int32_t>(pos.length));
                event_t sourceConsumed = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(sourceConsumed);
                WaitFlag<HardEvent::V_MTE2>(sourceConsumed);
            }

            event_t outputReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(outputReady);
            WaitFlag<HardEvent::V_MTE3>(outputReady);
            CopyOut(y_[outputOffset], accLocal, pos.length);

            event_t outputConsumed = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(outputConsumed);
            WaitFlag<HardEvent::MTE3_MTE2>(outputConsumed);
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    GlobalTensor<T> self_;
    GlobalTensor<T> source_;
    GlobalTensor<T> y_;
    GlobalTensor<int32_t> index_;
    uint64_t selfCount_;
    uint64_t indexCount_;
    uint64_t selfDim_;
    uint64_t inner_;
    uint64_t outer_;
    uint64_t chunksPerRow_;
    uint64_t taskCount_;
};

class KernelIndexAddInt8 {
public:
    __aicore__ inline void Init(
        GM_ADDR self,
        GM_ADDR index,
        GM_ADDR source,
        GM_ADDR y,
        const IndexAddTilingData &tiling)
    {
        self_.SetGlobalBuffer((__gm__ int8_t *)self);
        index_.SetGlobalBuffer((__gm__ int32_t *)index);
        source_.SetGlobalBuffer((__gm__ int8_t *)source);
        y_.SetGlobalBuffer((__gm__ int8_t *)y);

        selfCount_ = tiling.selfCount;
        indexCount_ = tiling.indexCount;
        selfDim_ = tiling.selfDim;
        inner_ = tiling.inner;
        const uint64_t denominator = selfDim_ * inner_;
        outer_ = denominator == 0 ? 0 : selfCount_ / denominator;
        chunksPerRow_ =
            inner_ == 0 ? 0 : (inner_ + kTileElements - 1) / kTileElements;
        taskCount_ = outer_ * selfDim_ * chunksPerRow_;

        pipe_.InitBuffer(indexBuf_, kMaxIndexCount * sizeof(int32_t));
        pipe_.InitBuffer(accRawBuf_, kTileElements * sizeof(int8_t));
        pipe_.InitBuffer(sourceRawBuf_, kTileElements * sizeof(int8_t));
        pipe_.InitBuffer(accHalfBuf_, kTileElements * sizeof(half));
        pipe_.InitBuffer(sourceHalfBuf_, kTileElements * sizeof(half));
        pipe_.InitBuffer(accInt16Buf_, kTileElements * sizeof(int16_t));
        pipe_.InitBuffer(sourceInt16Buf_, kTileElements * sizeof(int16_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        const bool indexFitsUb = indexCount_ <= kMaxIndexCount;
        if (indexFitsUb && indexCount_ > 0) {
            CopyIn(indexLocal, index_, static_cast<uint32_t>(indexCount_));
            event_t indexReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(indexReady);
            WaitFlag<HardEvent::MTE2_S>(indexReady);
        }

        LocalTensor<int8_t> accRaw = accRawBuf_.Get<int8_t>();
        LocalTensor<int8_t> sourceRaw = sourceRawBuf_.Get<int8_t>();
        LocalTensor<half> accHalf = accHalfBuf_.Get<half>();
        LocalTensor<half> sourceHalf = sourceHalfBuf_.Get<half>();
        LocalTensor<int16_t> accInt16 = accInt16Buf_.Get<int16_t>();
        LocalTensor<int16_t> sourceInt16 = sourceInt16Buf_.Get<int16_t>();
        const uint64_t blockCount = GetBlockNum();

        for (uint64_t task = GetBlockIdx(); task < taskCount_; task += blockCount) {
            const TaskPosition pos = DecodeTask(task, selfDim_, inner_);
            const uint64_t outputOffset =
                (pos.group * selfDim_ + pos.row) * inner_ + pos.column;

            CopyIn(accRaw, self_[outputOffset], pos.length);
            event_t selfReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(selfReady);
            WaitFlag<HardEvent::MTE2_V>(selfReady);
            Cast(accHalf, accRaw, RoundMode::CAST_NONE, pos.length);
            Cast(accInt16, accHalf, RoundMode::CAST_RINT, pos.length);

            for (uint64_t i = 0; i < indexCount_; ++i) {
                const int32_t rawIndex =
                    indexFitsUb ? indexLocal.GetValue(i) : index_.GetValue(i);
                const int64_t dstRow = NormalizeIndex(rawIndex, selfDim_);
                if (dstRow != static_cast<int64_t>(pos.row)) {
                    continue;
                }

                const uint64_t sourceOffset =
                    (pos.group * indexCount_ + i) * inner_ + pos.column;
                CopyIn(sourceRaw, source_[sourceOffset], pos.length);
                event_t sourceReady = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(sourceReady);
                WaitFlag<HardEvent::MTE2_V>(sourceReady);
                Cast(sourceHalf, sourceRaw, RoundMode::CAST_NONE, pos.length);
                Cast(sourceInt16, sourceHalf, RoundMode::CAST_RINT, pos.length);
                Add(accInt16, accInt16, sourceInt16, static_cast<int32_t>(pos.length));

                event_t sourceConsumed = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(sourceConsumed);
                WaitFlag<HardEvent::V_MTE2>(sourceConsumed);
            }

            event_t vectorToScalar = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(vectorToScalar);
            WaitFlag<HardEvent::V_S>(vectorToScalar);
            for (uint32_t i = 0; i < pos.length; ++i) {
                accRaw.SetValue(i, static_cast<int8_t>(accInt16.GetValue(i)));
            }

            event_t scalarToOutput = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(scalarToOutput);
            WaitFlag<HardEvent::S_MTE3>(scalarToOutput);
            CopyOut(y_[outputOffset], accRaw, pos.length);

            event_t outputConsumed = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(outputConsumed);
            WaitFlag<HardEvent::MTE3_S>(outputConsumed);
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> accRawBuf_;
    TBuf<TPosition::VECCALC> sourceRawBuf_;
    TBuf<TPosition::VECCALC> accHalfBuf_;
    TBuf<TPosition::VECCALC> sourceHalfBuf_;
    TBuf<TPosition::VECCALC> accInt16Buf_;
    TBuf<TPosition::VECCALC> sourceInt16Buf_;
    GlobalTensor<int8_t> self_;
    GlobalTensor<int8_t> source_;
    GlobalTensor<int8_t> y_;
    GlobalTensor<int32_t> index_;
    uint64_t selfCount_;
    uint64_t indexCount_;
    uint64_t selfDim_;
    uint64_t inner_;
    uint64_t outer_;
    uint64_t chunksPerRow_;
    uint64_t taskCount_;
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

        pipe_.InitBuffer(indexBuf_, kMaxIndexCount * sizeof(int32_t));
        pipe_.InitBuffer(accRawBuf_, kTileElements * sizeof(bfloat16_t));
        pipe_.InitBuffer(sourceRawBuf_, kTileElements * sizeof(bfloat16_t));
        pipe_.InitBuffer(accFloatBuf_, kTileElements * sizeof(float));
        pipe_.InitBuffer(sourceFloatBuf_, kTileElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        const bool indexFitsUb = indexCount_ <= kMaxIndexCount;
        if (indexFitsUb && indexCount_ > 0) {
            CopyIn(indexLocal, index_, static_cast<uint32_t>(indexCount_));
            event_t indexReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(indexReady);
            WaitFlag<HardEvent::MTE2_S>(indexReady);
        }

        LocalTensor<bfloat16_t> accRaw = accRawBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> sourceRaw = sourceRawBuf_.Get<bfloat16_t>();
        LocalTensor<float> accFloat = accFloatBuf_.Get<float>();
        LocalTensor<float> sourceFloat = sourceFloatBuf_.Get<float>();
        const uint64_t blockCount = GetBlockNum();

        for (uint64_t task = GetBlockIdx(); task < taskCount_; task += blockCount) {
            const TaskPosition pos = DecodeTask(task, selfDim_, inner_);
            const uint64_t outputOffset =
                (pos.group * selfDim_ + pos.row) * inner_ + pos.column;

            CopyIn(accRaw, self_[outputOffset], pos.length);
            event_t selfReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(selfReady);
            WaitFlag<HardEvent::MTE2_V>(selfReady);
            Cast(accFloat, accRaw, RoundMode::CAST_NONE, pos.length);

            for (uint64_t i = 0; i < indexCount_; ++i) {
                const int32_t rawIndex =
                    indexFitsUb ? indexLocal.GetValue(i) : index_.GetValue(i);
                const int64_t dstRow = NormalizeIndex(rawIndex, selfDim_);
                if (dstRow != static_cast<int64_t>(pos.row)) {
                    continue;
                }

                const uint64_t sourceOffset =
                    (pos.group * indexCount_ + i) * inner_ + pos.column;
                CopyIn(sourceRaw, source_[sourceOffset], pos.length);
                event_t sourceReady = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(sourceReady);
                WaitFlag<HardEvent::MTE2_V>(sourceReady);
                Cast(sourceFloat, sourceRaw, RoundMode::CAST_NONE, pos.length);
                Add(accFloat, accFloat, sourceFloat, static_cast<int32_t>(pos.length));
                Cast(accRaw, accFloat, RoundMode::CAST_RINT, pos.length);
                Cast(accFloat, accRaw, RoundMode::CAST_NONE, pos.length);

                event_t sourceConsumed = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(sourceConsumed);
                WaitFlag<HardEvent::V_MTE2>(sourceConsumed);
            }

            Cast(accRaw, accFloat, RoundMode::CAST_RINT, pos.length);
            event_t outputReady = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(outputReady);
            WaitFlag<HardEvent::V_MTE3>(outputReady);
            CopyOut(y_[outputOffset], accRaw, pos.length);

            event_t outputConsumed = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(outputConsumed);
            WaitFlag<HardEvent::MTE3_MTE2>(outputConsumed);
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> indexBuf_;
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
    KernelIndexAddVector<T> op;
    op.Init(self, index, source, y, tiling);
    op.Process();
}

template <>
__aicore__ inline void Run<int8_t>(
    GM_ADDR self,
    GM_ADDR index,
    GM_ADDR source,
    GM_ADDR y,
    const IndexAddTilingData &tiling)
{
    KernelIndexAddInt8 op;
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
