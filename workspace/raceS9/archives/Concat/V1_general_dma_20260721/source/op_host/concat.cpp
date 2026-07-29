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
    for (uint32_t i = 0; i < inputCount; ++i) {
        const auto &cur = context->GetDynamicInputShape(0, i)->GetStorageShape();
        if (cur.GetDimNum() != static_cast<size_t>(rank)) {
            return ge::GRAPH_FAILED;
        }
        prefixes[i] = outputAxis;
        axisSizes[i] = static_cast<uint64_t>(cur.GetDim(dim));
        outputAxis += axisSizes[i];
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCores = platform.GetCoreNum();
    const uint64_t tasks = outer * inputCount;
    const uint32_t blockDim = std::max<uint32_t>(1, std::min<uint64_t>(availableCores, tasks));

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t tileBytes = static_cast<uint32_t>(std::min<uint64_t>(64 * 1024, ubBytes / 2));

    ConcatTilingData tiling;
    tiling.set_inputCount(inputCount);
    tiling.set_dtypeCode(DtypeCode(context->GetDynamicInputDesc(0, 0)->GetDataType()));
    tiling.set_blockDim(blockDim);
    tiling.set_tileBytes(std::max<uint32_t>(32, tileBytes / 32 * 32));
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
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}  // namespace ops
