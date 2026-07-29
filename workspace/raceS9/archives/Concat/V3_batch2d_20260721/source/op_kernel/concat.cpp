#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

template <typename T>
class KernelConcat {
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
        tileElems_ = t.tileBytes / sizeof(T);
        pipe_.InitBuffer(buf_, t.tileBytes);
    }

    __aicore__ inline void Process(const ConcatTilingData &t)
    {
        ListTensorDesc list((__gm__ void *)inputList_);
        LocalTensor<T> local = buf_.Get<T>();
        const uint64_t taskCount = static_cast<uint64_t>(rowTaskCount_) * inputCount_;
        for (uint64_t task = GetBlockIdx(); task < taskCount; task += blockDim_) {
            const uint32_t inputId = static_cast<uint32_t>(task % inputCount_);
            const uint64_t rowTask = task / inputCount_;
            const uint64_t firstOuter = rowTask * rowsPerTask_;
            const uint32_t rowCount = static_cast<uint32_t>(
                (outer_ - firstOuter < rowsPerTask_) ? outer_ - firstOuter : rowsPerTask_);
            const uint64_t count = t.axisSizes[inputId] * inner_;
            if (count == 0) {
                continue;
            }
            GlobalTensor<T> src;
            src.SetGlobalBuffer((__gm__ T *)list.GetDataPtr<__gm__ uint8_t>(inputId));
            const uint64_t srcBase = firstOuter * count;
            const uint64_t dstBase = (firstOuter * outputAxis_ + t.axisPrefixes[inputId]) * inner_;
            const uint64_t segmentBytes = count * sizeof(T);

            if (segmentBytes <= t.tileBytes && segmentBytes * rowCount <= t.tileBytes) {
                const uint32_t blockBytes = static_cast<uint32_t>(segmentBytes);
                DataCopyExtParams inputParams{static_cast<uint16_t>(rowCount), blockBytes, 0, 0, 0};
                DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(local, src[srcBase], inputParams, pad);
                SyncInputToOutput();
                const uint64_t outputGap = (outputAxis_ * inner_ - count) * sizeof(T);
                DataCopyExtParams outputParams{
                    static_cast<uint16_t>(rowCount), blockBytes, 0,
                    static_cast<uint32_t>(outputGap), 0};
                DataCopyPad(out_[dstBase], local, outputParams);
                SyncOutputToInput();
                continue;
            }

            for (uint32_t row = 0; row < rowCount; ++row) {
                const uint64_t rowSrcBase = srcBase + static_cast<uint64_t>(row) * count;
                const uint64_t rowDstBase = dstBase + static_cast<uint64_t>(row) * outputAxis_ * inner_;
                for (uint64_t done = 0; done < count; done += tileElems_) {
                    const uint32_t cur = static_cast<uint32_t>(
                        (count - done < tileElems_) ? count - done : tileElems_);
                    DataCopyExtParams p{1, cur * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                    DataCopyPad(local, src[rowSrcBase + done], p, pad);
                    SyncInputToOutput();
                    DataCopyPad(out_[rowDstBase + done], local, p);
                    SyncOutputToInput();
                }
            }
        }
    }

private:
    __aicore__ inline void SyncInputToOutput()
    {
        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
    }

    __aicore__ inline void SyncOutputToInput()
    {
        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(event);
        WaitFlag<HardEvent::MTE3_MTE2>(event);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> buf_;
    GM_ADDR inputList_;
    GlobalTensor<T> out_;
    uint32_t inputCount_, blockDim_, rowsPerTask_, rowTaskCount_, tileElems_;
    uint64_t outer_, inner_, outputAxis_;
};

template <typename T>
__aicore__ inline void RunConcat(GM_ADDR inputs, GM_ADDR y, const ConcatTilingData &t)
{
    KernelConcat<T> op;
    op.Init(inputs, y, t);
    op.Process(t);
}

extern "C" __global__ __aicore__ void concat_custom(
    GM_ADDR inputs,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.dtypeCode == 0) {
        RunConcat<half>(inputs, y, tilingData);
    } else if (tilingData.dtypeCode == 1) {
        RunConcat<float>(inputs, y, tilingData);
    } else if (tilingData.dtypeCode == 2) {
        RunConcat<int32_t>(inputs, y, tilingData);
    } else {
        RunConcat<int8_t>(inputs, y, tilingData);
    }
}
