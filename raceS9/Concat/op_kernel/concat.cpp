#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t INLINE_INPUT_LIMIT = 128;
}

template <typename T>
class KernelConcat {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling)
    {
        inputs_ = inputs;
        output_.SetGlobalBuffer((__gm__ T *)output);
        inputCount_ = tiling.inputCount;
        outer_ = tiling.outer;
        inner_ = tiling.inner;
        outputAxis_ = tiling.outputAxis;
        iterations_ = tiling.iterations;
        iterationsPerInput_ = tiling.iterationsPerInput;
        rowsPerIteration_ = tiling.rowsPerIteration;
        axis_ = tiling.axis;
        tileElements_ = tiling.tileBytes / sizeof(T);
        pipe_.InitBuffer(copyQueue_, BUFFER_NUM, tiling.tileBytes);
    }

    __aicore__ inline void Process(const ConcatTilingData &tiling)
    {
        ListTensorDesc list((__gm__ void *)inputs_);
        const uint64_t iterationBegin = GetBlockIdx();
        const uint64_t iterationStep = GetBlockNum();
        const uint64_t iterationLoopStep = iterationStep * BUFFER_NUM;

        for (uint64_t base = iterationBegin; base < iterations_; base += iterationLoopStep) {
            for (uint32_t bufferId = 0; bufferId < BUFFER_NUM; ++bufferId) {
                const uint64_t iteration = base + bufferId * iterationStep;
                if (iteration >= iterations_) {
                    break;
                }
                ProcessIteration(list, tiling, iteration);
            }
        }
    }

private:
    __aicore__ inline void ProcessIteration(
        ListTensorDesc &list, const ConcatTilingData &tiling, uint64_t iteration)
    {
        const uint32_t inputId = static_cast<uint32_t>(iteration / iterationsPerInput_);
        const uint64_t inputIteration = iteration % iterationsPerInput_;
        const uint64_t firstOuter = inputIteration * rowsPerIteration_;
        const uint32_t rowCount = static_cast<uint32_t>(
            outer_ - firstOuter < rowsPerIteration_ ? outer_ - firstOuter : rowsPerIteration_);

        uint64_t axisSize = 0;
        uint64_t axisPrefix = 0;
        GetAxisInfo(list, tiling, inputId, axisSize, axisPrefix);
        const uint64_t segmentElements = axisSize * inner_;
        if (segmentElements == 0 || rowCount == 0) {
            return;
        }

        GlobalTensor<T> input;
        input.SetGlobalBuffer((__gm__ T *)list.GetDataPtr<__gm__ uint8_t>(inputId));
        const uint64_t inputBase = firstOuter * segmentElements;
        const uint64_t outputBase = (firstOuter * outputAxis_ + axisPrefix) * inner_;
        const uint64_t segmentBytes = segmentElements * sizeof(T);
        const uint64_t paddedSegmentBytes = (segmentBytes + 31) / 32 * 32;
        const uint64_t outputGapBytes = (outputAxis_ * inner_ - segmentElements) * sizeof(T);

        if (segmentBytes <= tiling.tileBytes &&
            paddedSegmentBytes * rowCount <= tiling.tileBytes &&
            outputGapBytes <= 0xFFFFFFFFULL) {
            CopyRows(input, inputBase, outputBase, rowCount,
                     static_cast<uint32_t>(segmentBytes),
                     static_cast<uint32_t>(outputGapBytes));
            return;
        }

        for (uint32_t row = 0; row < rowCount; ++row) {
            const uint64_t rowInput = inputBase + static_cast<uint64_t>(row) * segmentElements;
            const uint64_t rowOutput =
                outputBase + static_cast<uint64_t>(row) * outputAxis_ * inner_;
            for (uint64_t done = 0; done < segmentElements; done += tileElements_) {
                const uint32_t current = static_cast<uint32_t>(
                    segmentElements - done < tileElements_ ? segmentElements - done : tileElements_);
                CopyRows(input, rowInput + done, rowOutput + done, 1,
                         current * static_cast<uint32_t>(sizeof(T)), 0);
            }
        }
    }

    __aicore__ inline void CopyRows(
        GlobalTensor<T> &input, uint64_t inputOffset, uint64_t outputOffset,
        uint16_t rowCount, uint32_t blockBytes, uint32_t outputGapBytes)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        DataCopyExtParams inputParams{rowCount, blockBytes, 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, input[inputOffset], inputParams, pad);
        copyQueue_.EnQue(local);
        local = copyQueue_.DeQue<T>();
        DataCopyExtParams outputParams{rowCount, blockBytes, 0, outputGapBytes, 0};
        DataCopyPad(output_[outputOffset], local, outputParams);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void GetAxisInfo(
        ListTensorDesc &list, const ConcatTilingData &tiling, uint32_t inputId,
        uint64_t &axisSize, uint64_t &axisPrefix)
    {
        if (inputCount_ <= INLINE_INPUT_LIMIT) {
            axisSize = tiling.axisSizes[inputId];
            axisPrefix = tiling.axisPrefixes[inputId];
            return;
        }

        uint64_t shape[8] = {};
        TensorDesc<uint8_t> desc;
        desc.SetShapeAddr(shape);
        for (uint32_t i = 0; i <= inputId; ++i) {
            list.GetDesc(desc, i);
            const uint64_t current = desc.GetShape(axis_);
            if (i == inputId) {
                axisSize = current;
            } else {
                axisPrefix += current;
            }
        }
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue_;
    GM_ADDR inputs_;
    GlobalTensor<T> output_;
    uint32_t inputCount_, rowsPerIteration_, axis_, tileElements_;
    uint64_t outer_, inner_, outputAxis_, iterations_, iterationsPerInput_;
};

template <typename T>
__aicore__ inline void RunConcat(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling)
{
    KernelConcat<T> op;
    op.Init(inputs, output, tiling);
    op.Process(tiling);
}

extern "C" __global__ __aicore__ void concat(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.dtypeCode == 0) {
        RunConcat<half>(inputs, output, tilingData);
    } else if (tilingData.dtypeCode == 1) {
        RunConcat<float>(inputs, output, tilingData);
    } else if (tilingData.dtypeCode == 2) {
        RunConcat<int32_t>(inputs, output, tilingData);
    } else {
        RunConcat<int8_t>(inputs, output, tilingData);
    }
}
