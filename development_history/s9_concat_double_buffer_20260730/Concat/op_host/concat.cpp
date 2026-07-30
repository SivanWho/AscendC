#include "concat_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace {
bool DtypeCode(ge::DataType dt, uint32_t &code)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            code = 0;
            return true;
        case ge::DT_FLOAT:
            code = 1;
            return true;
        case ge::DT_INT32:
            code = 2;
            return true;
        case ge::DT_INT8:
            code = 3;
            return true;
        default:
            return false;
    }
}

uint32_t DtypeBytes(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint64_t AlignUp32(uint64_t value)
{
    return (value + 31) / 32 * 32;
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > UINT64_MAX / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const uint32_t inputCount = static_cast<uint32_t>(context->GetComputeNodeInputNum());
    if (inputCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetInt(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &shape = context->GetDynamicInputShape(0, 0)->GetStorageShape();
    const int32_t rank = static_cast<int32_t>(shape.GetDimNum());
    int64_t dim = *attrs->GetInt(0);
    if (dim < 0) {
        dim += rank;
    }
    if (rank <= 0 || rank > 8 || dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outer = 1, inner = 1;
    for (int32_t i = 0; i < dim; ++i) {
        const int64_t extent = shape.GetDim(i);
        if (extent < 0 || !CheckedMul(outer, static_cast<uint64_t>(extent), outer)) {
            return ge::GRAPH_FAILED;
        }
    }
    for (int32_t i = static_cast<int32_t>(dim) + 1; i < rank; ++i) {
        const int64_t extent = shape.GetDim(i);
        if (extent < 0 || !CheckedMul(inner, static_cast<uint64_t>(extent), inner)) {
            return ge::GRAPH_FAILED;
        }
    }

    const auto dtype = context->GetDynamicInputDesc(0, 0)->GetDataType();
    uint32_t dtypeCode = 0;
    if (!DtypeCode(dtype, dtypeCode)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t axisSizes[CONCAT_MAX_INPUTS] = {};
    uint64_t prefixes[CONCAT_MAX_INPUTS] = {};
    uint64_t outputAxis = 0;
    uint64_t maxAxisSize = 0;
    for (uint32_t i = 0; i < inputCount; ++i) {
        const auto &cur = context->GetDynamicInputShape(0, i)->GetStorageShape();
        if (cur.GetDimNum() != static_cast<size_t>(rank)) {
            return ge::GRAPH_FAILED;
        }
        if (context->GetDynamicInputDesc(0, i)->GetDataType() != dtype) {
            return ge::GRAPH_FAILED;
        }
        for (int32_t axis = 0; axis < rank; ++axis) {
            const int64_t extent = cur.GetDim(axis);
            if (extent < 0 || (axis != dim && extent != shape.GetDim(axis))) {
                return ge::GRAPH_FAILED;
            }
        }
        const uint64_t axisSize = static_cast<uint64_t>(cur.GetDim(dim));
        if (i < CONCAT_MAX_INPUTS) {
            prefixes[i] = outputAxis;
            axisSizes[i] = axisSize;
        }
        maxAxisSize = std::max(maxAxisSize, axisSize);
        if (axisSize > UINT64_MAX - outputAxis) {
            return ge::GRAPH_FAILED;
        }
        outputAxis += axisSize;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCores = std::max<uint32_t>(1, platform.GetCoreNum());

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t tileBytes = static_cast<uint32_t>(std::min<uint64_t>(64 * 1024, ubBytes / 2));
    const uint32_t elemBytes = DtypeBytes(dtype);
    if (elemBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    // Each task batches several outer rows for one input into one 2-D DMA.  Cap
    // the batch both by UB capacity and by the amount of parallelism available.
    uint64_t maxSegmentElems = 0;
    uint64_t maxSegmentBytes = 0;
    uint64_t outputElems = 0;
    uint64_t inputBytes = 0;
    if (!CheckedMul(maxAxisSize, inner, maxSegmentElems) ||
        !CheckedMul(maxSegmentElems, elemBytes, maxSegmentBytes) ||
        !CheckedMul(outer, outputAxis, outputElems) ||
        !CheckedMul(outputElems, inner, outputElems) ||
        !CheckedMul(outputElems, elemBytes, inputBytes)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t maxPaddedSegmentBytes =
        std::max<uint64_t>(1, AlignUp32(maxSegmentBytes));
    const uint64_t rowsForUb =
        std::max<uint64_t>(1, tileBytes / maxPaddedSegmentBytes);
    constexpr uint64_t BYTES_PER_CORE = 16 * 1024;
    const uint64_t targetCores = std::min<uint64_t>(
        availableCores, std::max<uint64_t>(1, CeilDiv(inputBytes, BYTES_PER_CORE)));
    const uint64_t chunksPerInput = std::max<uint64_t>(1, CeilDiv(targetCores, inputCount));
    const uint64_t rowsForParallel = outer == 0 ? 1 : CeilDiv(outer, chunksPerInput);
    const uint32_t rowsPerTask = static_cast<uint32_t>(
        std::min<uint64_t>(4095, std::max<uint64_t>(1, std::min(rowsForUb, rowsForParallel))));
    const uint64_t rowTaskCount64 = outer == 0 ? 0 : CeilDiv(outer, rowsPerTask);
    if (rowTaskCount64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t rowTaskCount = static_cast<uint32_t>(rowTaskCount64);
    const uint64_t tasks = static_cast<uint64_t>(rowTaskCount) * inputCount;
    const uint32_t blockDim = std::max<uint32_t>(1, std::min<uint64_t>(availableCores, tasks));

    ConcatTilingData tiling;
    tiling.set_inputCount(inputCount);
    tiling.set_dtypeCode(dtypeCode);
    tiling.set_blockDim(blockDim);
    tiling.set_tileBytes(std::max<uint32_t>(32, tileBytes / 32 * 32));
    tiling.set_rowsPerTask(rowsPerTask);
    tiling.set_rowTaskCount(rowTaskCount);
    tiling.set_axis(static_cast<uint32_t>(dim));
    tiling.set_outer(outer);
    tiling.set_inner(inner);
    tiling.set_outputAxis(outputAxis);
    tiling.set_axisSizes(axisSizes);
    tiling.set_axisPrefixes(prefixes);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *attrs = context->GetAttrs();
    const gert::Shape *first = context->GetDynamicInputShape(0, 0);
    if (attrs == nullptr || first == nullptr || attrs->GetInt(0) == nullptr) {
        return GRAPH_FAILED;
    }
    gert::Shape *out = context->GetOutputShape(0);
    *out = *first;
    int64_t dim = *attrs->GetInt(0);
    const int32_t rank = static_cast<int32_t>(first->GetDimNum());
    if (dim < 0) {
        dim += rank;
    }
    if (rank <= 0 || rank > 8 || dim < 0 || dim >= rank) {
        return GRAPH_FAILED;
    }
    int64_t sum = 0;
    const size_t count = context->GetComputeNodeInputNum();
    for (size_t i = 0; i < count; ++i) {
        const gert::Shape *current = context->GetDynamicInputShape(0, i);
        if (current == nullptr || current->GetDimNum() != first->GetDimNum()) {
            return GRAPH_FAILED;
        }
        for (int32_t axis = 0; axis < rank; ++axis) {
            const int64_t extent = current->GetDim(axis);
            if (extent < 0 || (axis != dim && extent != first->GetDim(axis))) {
                return GRAPH_FAILED;
            }
        }
        const int64_t axisSize = current->GetDim(dim);
        if (axisSize > INT64_MAX - sum) {
            return GRAPH_FAILED;
        }
        sum += axisSize;
    }
    out->SetDim(dim, sum);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetDynamicInputDataType(0, 0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char *name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "concat");
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(Concat);
}  // namespace ops
