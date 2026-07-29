#include "transpose_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>

namespace {
uint32_t Code(ge::DataType d)
{
    return d == ge::DT_FLOAT16 ? 0 : d == ge::DT_FLOAT ? 1 : d == ge::DT_INT32 ? 2 : 3;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *ctx)
{
    const auto &s = ctx->GetInputShape(0)->GetStorageShape();
    const uint32_t rank = s.GetDimNum();
    const auto *attrs = ctx->GetAttrs();
    if (rank == 0 || rank > 8 || !attrs || !attrs->GetListInt(0)) {
        return ge::GRAPH_FAILED;
    }
    const auto *perm = attrs->GetListInt(0);
    if (perm->GetSize() != rank) {
        return ge::GRAPH_FAILED;
    }
    uint64_t inStride[8] = {}, outShape[8] = {}, mapped[8] = {};
    bool seen[8] = {};
    uint64_t stride = 1, total = 1;
    for (int32_t i = rank - 1; i >= 0; --i) {
        inStride[i] = stride;
        stride *= s.GetDim(i);
    }
    for (uint32_t i = 0; i < rank; ++i) {
        int64_t p = perm->GetData()[i];
        if (p < 0) {
            p += rank;
        }
        if (p < 0 || p >= rank || seen[p]) {
            return ge::GRAPH_FAILED;
        }
        seen[p] = true;
        outShape[i] = s.GetDim(p);
        mapped[i] = inStride[p];
        total *= outShape[i];
    }
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint32_t blocks = std::max<uint32_t>(1, std::min<uint64_t>(platform.GetCoreNum(), total));
    TransposeTilingData t;
    t.set_dtypeCode(Code(ctx->GetInputDesc(0)->GetDataType()));
    t.set_rank(rank);
    t.set_blockDim(blocks);
    t.set_outputCount(total);
    t.set_outputShape(outShape);
    t.set_sourceStrideForOutputDim(mapped);
    t.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(t.GetDataSize());
    ctx->SetBlockDim(blocks);
    ctx->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx)
{
    const auto *x = ctx->GetInputShape(0);
    const auto *attrs = ctx->GetAttrs();
    if (!attrs || !attrs->GetListInt(0)) {
        return GRAPH_FAILED;
    }
    const auto *p = attrs->GetListInt(0);
    auto *y = ctx->GetOutputShape(0);
    y->SetDimNum(p->GetSize());
    for (size_t i = 0; i < p->GetSize(); ++i) {
        int64_t d = p->GetData()[i];
        if (d < 0) {
            d += x->GetDimNum();
        }
        y->SetDim(i, x->GetDim(d));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferType(gert::InferDataTypeContext *ctx)
{
    ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char *n) : OpDef(n)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dims").AttrType(REQUIRED).ListInt();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);
}  // namespace ops
