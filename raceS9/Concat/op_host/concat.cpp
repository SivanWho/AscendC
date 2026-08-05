#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace {
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t PRESERVED_UB_BYTES = 1024;
constexpr uint64_t MAX_TILE_BYTES = 64 * 1024;
constexpr uint64_t MAX_MTE_ROWS = 4095;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint64_t AlignUp32(uint64_t value)
{
    return (value + 31) / 32 * 32;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (rhs > UINT64_MAX - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > UINT64_MAX / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool DtypeInfo(ge::DataType dtype, uint32_t &code, uint32_t &bytes)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            code = 0;
            bytes = 2;
            return true;
        case ge::DT_FLOAT:
            code = 1;
            bytes = 4;
            return true;
        case ge::DT_INT32:
            code = 2;
            bytes = 4;
            return true;
        case ge::DT_INT8:
            code = 3;
            bytes = 1;
            return true;
        default:
            return false;
    }
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const uint32_t inputCount = static_cast<uint32_t>(context->GetComputeNodeInputNum());
    const auto *attrs = context->GetAttrs();
    if (inputCount == 0 || attrs == nullptr || attrs->GetInt(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &firstShape = context->GetDynamicInputShape(0, 0)->GetStorageShape();
    const int32_t rank = static_cast<int32_t>(firstShape.GetDimNum());
    int64_t axis = *attrs->GetInt(0);
    if (axis < 0) {
        axis += rank;
    }
    if (rank <= 0 || rank > 8 || axis < 0 || axis >= rank) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outer = 1;
    uint64_t inner = 1;
    for (int32_t i = 0; i < axis; ++i) {
        const int64_t extent = firstShape.GetDim(i);
        if (extent < 0 || !CheckedMul(outer, static_cast<uint64_t>(extent), outer)) {
            return ge::GRAPH_FAILED;
        }
    }
    for (int32_t i = static_cast<int32_t>(axis) + 1; i < rank; ++i) {
        const int64_t extent = firstShape.GetDim(i);
        if (extent < 0 || !CheckedMul(inner, static_cast<uint64_t>(extent), inner)) {
            return ge::GRAPH_FAILED;
        }
    }

    const ge::DataType dtype = context->GetDynamicInputDesc(0, 0)->GetDataType();
    uint32_t dtypeCode = 0;
    uint32_t elementBytes = 0;
    if (!DtypeInfo(dtype, dtypeCode, elementBytes)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t axisSizes[CONCAT_MAX_INPUTS] = {};
    uint64_t axisPrefixes[CONCAT_MAX_INPUTS] = {};
    uint64_t outputAxis = 0;
    uint64_t maxAxisSize = 0;
    for (uint32_t inputId = 0; inputId < inputCount; ++inputId) {
        const auto &shape = context->GetDynamicInputShape(0, inputId)->GetStorageShape();
        if (shape.GetDimNum() != static_cast<size_t>(rank) ||
            context->GetDynamicInputDesc(0, inputId)->GetDataType() != dtype) {
            return ge::GRAPH_FAILED;
        }
        for (int32_t dim = 0; dim < rank; ++dim) {
            const int64_t extent = shape.GetDim(dim);
            if (extent < 0 || (dim != axis && extent != firstShape.GetDim(dim))) {
                return ge::GRAPH_FAILED;
            }
        }

        const uint64_t axisSize = static_cast<uint64_t>(shape.GetDim(axis));
        if (inputId < CONCAT_MAX_INPUTS) {
            axisSizes[inputId] = axisSize;
            axisPrefixes[inputId] = outputAxis;
        }
        maxAxisSize = std::max(maxAxisSize, axisSize);
        if (!CheckedAdd(outputAxis, axisSize, outputAxis)) {
            return ge::GRAPH_FAILED;
        }
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (ubBytes <= PRESERVED_UB_BYTES + BUFFER_NUM * 32) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t bytesPerBuffer = (ubBytes - PRESERVED_UB_BYTES) / BUFFER_NUM;
    const uint32_t tileBytes = static_cast<uint32_t>(
        std::max<uint64_t>(32, std::min(MAX_TILE_BYTES, bytesPerBuffer) / 32 * 32));

    uint64_t maxSegmentElements = 0;
    uint64_t maxSegmentBytes = 0;
    uint64_t outputElements = 0;
    uint64_t outputBytes = 0;
    if (!CheckedMul(maxAxisSize, inner, maxSegmentElements) ||
        !CheckedMul(maxSegmentElements, elementBytes, maxSegmentBytes) ||
        !CheckedMul(outputAxis, inner, outputElements) ||
        !CheckedMul(outputElements, elementBytes, outputBytes) ||
        maxSegmentBytes > UINT64_MAX - 31) {
        return ge::GRAPH_FAILED;
    }
    static_cast<void>(outputBytes);
    const uint64_t paddedSegmentBytes = std::max<uint64_t>(32, AlignUp32(maxSegmentBytes));
    const uint32_t rowsPerIteration = static_cast<uint32_t>(std::min<uint64_t>(
        MAX_MTE_ROWS, std::max<uint64_t>(1, tileBytes / paddedSegmentBytes)));
    const uint64_t iterationsPerInput = outer == 0 ? 0 : CeilDiv(outer, rowsPerIteration);
    uint64_t iterations = 0;
    if (!CheckedMul(iterationsPerInput, inputCount, iterations)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t availableCores = std::max<uint32_t>(1, platform.GetCoreNumAiv());
    const uint32_t blockDim = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(availableCores, iterations)));

    ConcatTilingData tiling;
    tiling.set_inputCount(inputCount);
    tiling.set_dtypeCode(dtypeCode);
    tiling.set_tileBytes(tileBytes);
    tiling.set_rowsPerIteration(rowsPerIteration);
    tiling.set_axis(static_cast<uint32_t>(axis));
    tiling.set_iterations(iterations);
    tiling.set_iterationsPerInput(iterationsPerInput);
    tiling.set_outer(outer);
    tiling.set_inner(inner);
    tiling.set_outputAxis(outputAxis);
    tiling.set_axisSizes(axisSizes);
    tiling.set_axisPrefixes(axisPrefixes);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
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

    const int32_t rank = static_cast<int32_t>(first->GetDimNum());
    int64_t axis = *attrs->GetInt(0);
    if (axis < 0) {
        axis += rank;
    }
    if (rank <= 0 || rank > 8 || axis < 0 || axis >= rank) {
        return GRAPH_FAILED;
    }

    int64_t outputAxis = 0;
    const size_t inputCount = context->GetComputeNodeInputNum();
    for (size_t inputId = 0; inputId < inputCount; ++inputId) {
        const gert::Shape *shape = context->GetDynamicInputShape(0, inputId);
        if (shape == nullptr || shape->GetDimNum() != first->GetDimNum()) {
            return GRAPH_FAILED;
        }
        for (int32_t dim = 0; dim < rank; ++dim) {
            const int64_t extent = shape->GetDim(dim);
            if (extent < 0 || (dim != axis && extent != first->GetDim(dim))) {
                return GRAPH_FAILED;
            }
        }
        const int64_t axisSize = shape->GetDim(axis);
        if (axisSize > INT64_MAX - outputAxis) {
            return GRAPH_FAILED;
        }
        outputAxis += axisSize;
    }

    gert::Shape *output = context->GetOutputShape(0);
    *output = *first;
    output->SetDim(axis, outputAxis);
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
