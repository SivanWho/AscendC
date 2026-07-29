#include "concat_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace {
uint32_t DtypeCode(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return 0;
        case ge::DT_FLOAT:
            return 1;
        case ge::DT_INT32:
            return 2;
        case ge::DT_INT8:
            return 3;
        default:
            return 0;
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
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const uint32_t inputCount = static_cast<uint32_t>(context->GetComputeNodeInputNum());
    if (inputCount == 0 || inputCount > CONCAT_MAX_INPUTS) {
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
    if (rank <= 0 || dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outer = 1, inner = 1;
    for (int32_t i = 0; i < dim; ++i) {
        outer *= static_cast<uint64_t>(shape.GetDim(i));
    }
    for (int32_t i = static_cast<int32_t>(dim) + 1; i < rank; ++i) {
        inner *= static_cast<uint64_t>(shape.GetDim(i));
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
        prefixes[i] = outputAxis;
        axisSizes[i] = static_cast<uint64_t>(cur.GetDim(dim));
        maxAxisSize = std::max(maxAxisSize, axisSizes[i]);
        outputAxis += axisSizes[i];
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCores = std::max<uint32_t>(1, platform.GetCoreNum());

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t tileBytes = static_cast<uint32_t>(std::min<uint64_t>(64 * 1024, ubBytes / 2));
    const auto dtype = context->GetDynamicInputDesc(0, 0)->GetDataType();
    const uint32_t elemBytes = DtypeBytes(dtype);
    if (elemBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    // Each task batches several outer rows for one input into one 2-D DMA.  Cap
    // the batch both by UB capacity and by the amount of parallelism available.
    const uint64_t maxSegmentBytes = std::max<uint64_t>(1, maxAxisSize * inner * elemBytes);
    const uint64_t rowsForUb = std::max<uint64_t>(1, tileBytes / maxSegmentBytes);
    constexpr uint64_t BYTES_PER_CORE = 16 * 1024;
    const uint64_t inputBytes = outer * outputAxis * inner * elemBytes;
    const uint64_t targetCores = std::min<uint64_t>(
        availableCores, std::max<uint64_t>(1, CeilDiv(inputBytes, BYTES_PER_CORE)));
    const uint64_t chunksPerInput = std::max<uint64_t>(1, CeilDiv(targetCores, inputCount));
    const uint64_t rowsForParallel = outer == 0 ? 1 : CeilDiv(outer, chunksPerInput);
    const uint32_t rowsPerTask = static_cast<uint32_t>(
        std::min<uint64_t>(65535, std::max<uint64_t>(1, std::min(rowsForUb, rowsForParallel))));
    const uint32_t rowTaskCount = outer == 0 ? 0 : static_cast<uint32_t>(CeilDiv(outer, rowsPerTask));
    const uint64_t tasks = static_cast<uint64_t>(rowTaskCount) * inputCount;
    const uint32_t blockDim = std::max<uint32_t>(1, std::min<uint64_t>(availableCores, tasks));

    ConcatTilingData tiling;
    tiling.set_inputCount(inputCount);
    tiling.set_dtypeCode(DtypeCode(dtype));
    tiling.set_blockDim(blockDim);
    tiling.set_tileBytes(std::max<uint32_t>(32, tileBytes / 32 * 32));
    tiling.set_rowsPerTask(rowsPerTask);
    tiling.set_rowTaskCount(rowTaskCount);
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
    int64_t sum = 0;
    const size_t count = context->GetComputeNodeInputNum();
    for (size_t i = 0; i < count; ++i) {
        sum += context->GetDynamicInputShape(0, i)->GetDim(dim);
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
