#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t kInlineInputLimit = 128;
}

template <typename T>
class KernelConcatSmall {
public:
    __aicore__ inline void Init(GM_ADDR inputList, GM_ADDR output, const ConcatTilingData &t)
    {
        inputList_ = inputList;
        out_.SetGlobalBuffer((__gm__ T *)output);
        inputCount_ = t.inputCount;
        blockDim_ = t.blockDim;
        outer_ = t.outer;
        inner_ = t.inner;
        outputAxis_ = t.outputAxis;
        rowsPerTask_ = t.rowsPerTask;
        rowTaskCount_ = t.rowTaskCount;
        axis_ = t.axis;
        tileElems_ = t.tileBytes / sizeof(T);
        pipe_.InitBuffer(copyBuf_, t.tileBytes);
    }

    __aicore__ inline void Process(const ConcatTilingData &t)
    {
        ListTensorDesc list((__gm__ void *)inputList_);
        LocalTensor<T> local = copyBuf_.Get<T>();
        const uint64_t taskCount = static_cast<uint64_t>(rowTaskCount_) * inputCount_;
        for (uint64_t task = GetBlockIdx(); task < taskCount; task += blockDim_) {
            const uint32_t inputId = static_cast<uint32_t>(task % inputCount_);
            const uint64_t rowTask = task / inputCount_;
            const uint64_t firstOuter = rowTask * rowsPerTask_;
            const uint32_t rowCount = static_cast<uint32_t>(
                (outer_ - firstOuter < rowsPerTask_) ? outer_ - firstOuter : rowsPerTask_);

            uint64_t axisSize = 0;
            uint64_t axisPrefix = 0;
            GetAxisInfo(list, t, inputId, axisSize, axisPrefix);
            const uint64_t count = axisSize * inner_;
            if (count == 0) {
                continue;
            }

            GlobalTensor<T> src;
            src.SetGlobalBuffer((__gm__ T *)list.GetDataPtr<__gm__ uint8_t>(inputId));
            const uint64_t srcBase = firstOuter * count;
            const uint64_t dstBase = (firstOuter * outputAxis_ + axisPrefix) * inner_;
            const uint64_t segmentBytes = count * sizeof(T);
            const uint64_t paddedSegmentBytes = (segmentBytes + 31) / 32 * 32;
            const uint64_t outputGap = (outputAxis_ * inner_ - count) * sizeof(T);

            if (rowCount <= 4095 && segmentBytes <= t.tileBytes &&
                paddedSegmentBytes * rowCount <= t.tileBytes &&
                outputGap <= 0xFFFFFFFFULL) {
                const uint32_t blockBytes = static_cast<uint32_t>(segmentBytes);
                DataCopyExtParams inputParams{
                    static_cast<uint16_t>(rowCount), blockBytes, 0, 0, 0};
                DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(local, src[srcBase], inputParams, pad);
                SyncInputToOutput();
                DataCopyExtParams outputParams{
                    static_cast<uint16_t>(rowCount), blockBytes, 0,
                    static_cast<uint32_t>(outputGap), 0};
                DataCopyPad(out_[dstBase], local, outputParams);
                SyncOutputToInput();
                continue;
            }

            for (uint32_t row = 0; row < rowCount; ++row) {
                const uint64_t rowSrcBase = srcBase + static_cast<uint64_t>(row) * count;
                const uint64_t rowDstBase =
                    dstBase + static_cast<uint64_t>(row) * outputAxis_ * inner_;
                for (uint64_t done = 0; done < count; done += tileElems_) {
                    const uint32_t cur = static_cast<uint32_t>(
                        (count - done < tileElems_) ? count - done : tileElems_);
                    DataCopyExtParams params{
                        1, cur * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                    DataCopyPad(local, src[rowSrcBase + done], params, pad);
                    SyncInputToOutput();
                    DataCopyPad(out_[rowDstBase + done], local, params);
                    SyncOutputToInput();
                }
            }
        }
    }

private:
    __aicore__ inline void GetAxisInfo(
        ListTensorDesc &list,
        const ConcatTilingData &t,
        uint32_t inputId,
        uint64_t &axisSize,
        uint64_t &axisPrefix)
    {
        if (inputCount_ <= kInlineInputLimit) {
            axisSize = t.axisSizes[inputId];
            axisPrefix = t.axisPrefixes[inputId];
            return;
        }

        uint64_t descShape[8] = {};
        TensorDesc<uint8_t> desc;
        desc.SetShapeAddr(descShape);
        axisPrefix = 0;
        for (uint32_t i = 0; i <= inputId; ++i) {
            list.GetDesc(desc, i);
            const uint64_t currentAxisSize = desc.GetShape(axis_);
            if (i == inputId) {
                axisSize = currentAxisSize;
            } else {
                axisPrefix += currentAxisSize;
            }
        }
    }

    __aicore__ inline void SyncInputToOutput()
    {
        event_t event =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
    }

    __aicore__ inline void SyncOutputToInput()
    {
        event_t event =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(event);
        WaitFlag<HardEvent::MTE3_MTE2>(event);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> copyBuf_;
    GM_ADDR inputList_;
    GlobalTensor<T> out_;
    uint32_t inputCount_, blockDim_, rowsPerTask_, rowTaskCount_, axis_, tileElems_;
    uint64_t outer_, inner_, outputAxis_;
};

template <typename T>
class KernelConcatLongSerial {
public:
    __aicore__ inline void Init(GM_ADDR inputList, GM_ADDR output, const ConcatTilingData &t)
    {
        inputList_ = inputList;
        out_.SetGlobalBuffer((__gm__ T *)output);
        inputCount_ = t.inputCount;
        blockDim_ = t.blockDim;
        outer_ = t.outer;
        inner_ = t.inner;
        outputAxis_ = t.outputAxis;
        rowsPerTask_ = t.rowsPerTask;
        rowTaskCount_ = t.rowTaskCount;
        axis_ = t.axis;
        tileElems_ = t.tileBytes / sizeof(T);
        pipe_.InitBuffer(copyQue_, 2, t.tileBytes);
    }

    __aicore__ inline void Process(const ConcatTilingData &t)
    {
        ListTensorDesc list((__gm__ void *)inputList_);
        const uint64_t taskCount = static_cast<uint64_t>(rowTaskCount_) * inputCount_;
        for (uint64_t task = GetBlockIdx(); task < taskCount; task += blockDim_) {
            const uint32_t inputId = static_cast<uint32_t>(task % inputCount_);
            const uint64_t rowTask = task / inputCount_;
            const uint64_t firstOuter = rowTask * rowsPerTask_;
            const uint32_t rowCount = static_cast<uint32_t>(
                (outer_ - firstOuter < rowsPerTask_) ? outer_ - firstOuter : rowsPerTask_);
            uint64_t axisSize = 0;
            uint64_t axisPrefix = 0;
            GetAxisInfo(list, t, inputId, axisSize, axisPrefix);
            const uint64_t count = axisSize * inner_;
            if (count == 0) {
                continue;
            }

            GlobalTensor<T> src;
            src.SetGlobalBuffer((__gm__ T *)list.GetDataPtr<__gm__ uint8_t>(inputId));
            const uint64_t srcBase = firstOuter * count;
            const uint64_t dstBase = (firstOuter * outputAxis_ + axisPrefix) * inner_;
            for (uint32_t row = 0; row < rowCount; ++row) {
                const uint64_t rowSrcBase = srcBase + static_cast<uint64_t>(row) * count;
                const uint64_t rowDstBase =
                    dstBase + static_cast<uint64_t>(row) * outputAxis_ * inner_;
                for (uint64_t done = 0; done < count; done += tileElems_) {
                    const uint32_t cur = static_cast<uint32_t>(
                        (count - done < tileElems_) ? count - done : tileElems_);
                    DataCopyExtParams params{
                        1, cur * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                    LocalTensor<T> local = copyQue_.AllocTensor<T>();
                    DataCopyPad(local, src[rowSrcBase + done], params, pad);
                    copyQue_.EnQue(local);
                    local = copyQue_.DeQue<T>();
                    DataCopyPad(out_[rowDstBase + done], local, params);
                    copyQue_.FreeTensor(local);
                }
            }
        }
    }

private:
    __aicore__ inline void GetAxisInfo(
        ListTensorDesc &list,
        const ConcatTilingData &t,
        uint32_t inputId,
        uint64_t &axisSize,
        uint64_t &axisPrefix)
    {
        if (inputCount_ <= kInlineInputLimit) {
            axisSize = t.axisSizes[inputId];
            axisPrefix = t.axisPrefixes[inputId];
            return;
        }

        uint64_t descShape[8] = {};
        TensorDesc<uint8_t> desc;
        desc.SetShapeAddr(descShape);
        axisPrefix = 0;
        for (uint32_t i = 0; i <= inputId; ++i) {
            list.GetDesc(desc, i);
            const uint64_t currentAxisSize = desc.GetShape(axis_);
            if (i == inputId) {
                axisSize = currentAxisSize;
            } else {
                axisPrefix += currentAxisSize;
            }
        }
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQue_;
    GM_ADDR inputList_;
    GlobalTensor<T> out_;
    uint32_t inputCount_, blockDim_, rowsPerTask_, rowTaskCount_, axis_, tileElems_;
    uint64_t outer_, inner_, outputAxis_;
};

template <typename T>
class KernelConcatLongParallel {
public:
    __aicore__ inline void Init(GM_ADDR inputList, GM_ADDR output, const ConcatTilingData &t)
    {
        inputList_ = inputList;
        out_.SetGlobalBuffer((__gm__ T *)output);
        inputCount_ = t.inputCount;
        blockDim_ = t.blockDim;
        outer_ = t.outer;
        inner_ = t.inner;
        outputAxis_ = t.outputAxis;
        axis_ = t.axis;
        tileElems_ = t.tileBytes / sizeof(T);
        pipe_.InitBuffer(copyQue_, 2, t.tileBytes);
    }

    __aicore__ inline void Process(const ConcatTilingData &t)
    {
        ListTensorDesc list((__gm__ void *)inputList_);
        uint64_t globalTaskBase = 0;
        uint64_t scannedAxisPrefix = 0;
        uint64_t descShape[8] = {};
        TensorDesc<uint8_t> desc;
        desc.SetShapeAddr(descShape);

        for (uint32_t inputId = 0; inputId < inputCount_; ++inputId) {
            uint64_t axisSize = 0;
            uint64_t axisPrefix = 0;
            if (inputCount_ <= kInlineInputLimit) {
                axisSize = t.axisSizes[inputId];
                axisPrefix = t.axisPrefixes[inputId];
            } else {
                list.GetDesc(desc, inputId);
                axisSize = desc.GetShape(axis_);
                axisPrefix = scannedAxisPrefix;
                scannedAxisPrefix += axisSize;
            }

            const uint64_t count = axisSize * inner_;
            if (count == 0) {
                continue;
            }
            const uint64_t tilesPerRow = (count + tileElems_ - 1) / tileElems_;
            const uint64_t inputTaskCount = outer_ * tilesPerRow;
            const uint64_t baseResidue = globalTaskBase % blockDim_;
            const uint64_t firstRelative =
                (static_cast<uint64_t>(GetBlockIdx()) + blockDim_ - baseResidue) % blockDim_;

            GlobalTensor<T> src;
            src.SetGlobalBuffer((__gm__ T *)list.GetDataPtr<__gm__ uint8_t>(inputId));
            for (uint64_t relativeTask = firstRelative;
                 relativeTask < inputTaskCount;
                 relativeTask += blockDim_) {
                const uint64_t row = relativeTask / tilesPerRow;
                const uint64_t tile = relativeTask - row * tilesPerRow;
                const uint64_t done = tile * tileElems_;
                const uint32_t cur = static_cast<uint32_t>(
                    (count - done < tileElems_) ? count - done : tileElems_);
                const uint64_t srcBase = row * count + done;
                const uint64_t dstBase =
                    (row * outputAxis_ + axisPrefix) * inner_ + done;
                DataCopyExtParams params{
                    1, cur * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                LocalTensor<T> local = copyQue_.AllocTensor<T>();
                DataCopyPad(local, src[srcBase], params, pad);
                copyQue_.EnQue(local);
                local = copyQue_.DeQue<T>();
                DataCopyPad(out_[dstBase], local, params);
                copyQue_.FreeTensor(local);
            }
            globalTaskBase += inputTaskCount;
        }
    }

private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQue_;
    GM_ADDR inputList_;
    GlobalTensor<T> out_;
    uint32_t inputCount_, blockDim_, axis_, tileElems_;
    uint64_t outer_, inner_, outputAxis_;
};

template <typename T, uint32_t Path>
__aicore__ inline void RunConcat(GM_ADDR inputs, GM_ADDR y, const ConcatTilingData &t)
{
    if constexpr (Path == 1) {
        KernelConcatLongParallel<T> op;
        op.Init(inputs, y, t);
        op.Process(t);
    } else if constexpr (Path == 2) {
        KernelConcatLongSerial<T> op;
        op.Init(inputs, y, t);
        op.Process(t);
    } else {
        KernelConcatSmall<T> op;
        op.Init(inputs, y, t);
        op.Process(t);
    }
}

template <uint32_t Path>
__aicore__ inline void DispatchConcat(GM_ADDR inputs, GM_ADDR y, const ConcatTilingData &t)
{
    if (t.dtypeCode == 0) {
        RunConcat<half, Path>(inputs, y, t);
    } else if (t.dtypeCode == 1) {
        RunConcat<float, Path>(inputs, y, t);
    } else if (t.dtypeCode == 2) {
        RunConcat<int32_t, Path>(inputs, y, t);
    } else {
        RunConcat<int8_t, Path>(inputs, y, t);
    }
}

extern "C" __global__ __aicore__ void concat(
    GM_ADDR inputs,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        DispatchConcat<1>(inputs, y, tilingData);
    } else if (TILING_KEY_IS(2)) {
        DispatchConcat<2>(inputs, y, tilingData);
    } else if (TILING_KEY_IS(0)) {
        DispatchConcat<0>(inputs, y, tilingData);
    }
}
