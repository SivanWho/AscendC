#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace {
uint32_t Code(ge::DataType d)
{
    return d == ge::DT_FLOAT16 ? 0 : d == ge::DT_FLOAT ? 1 : d == ge::DT_INT32 ? 2 : d == ge::DT_INT8 ? 3 : 4;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *ctx)
{
    const auto &s = ctx->GetInputShape(0)->GetStorageShape();
    uint32_t rank = s.GetDimNum();
    auto *a = ctx->GetAttrs();
    if (rank == 0 || rank > 8 || !a || !a->GetInt(0)) {
        return ge::GRAPH_FAILED;
    }
    int64_t dim = *a->GetInt(0);
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }
    uint64_t inner = 1;
    for (uint32_t d = dim + 1; d < rank; ++d) {
        inner *= s.GetDim(d);
    }
    IndexAddTilingData t;
    t.set_dtypeCode(Code(ctx->GetInputDesc(0)->GetDataType()));
    t.set_rank(rank);
    t.set_selfCount(s.GetShapeSize());
    t.set_sourceCount(ctx->GetInputShape(2)->GetStorageShape().GetShapeSize());
    t.set_indexCount(ctx->GetInputShape(1)->GetStorageShape().GetShapeSize());
    t.set_selfDim(s.GetDim(dim));
    t.set_inner(inner);
    t.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(t.GetDataSize());
    // The safe kernel updates output rows in index order.  One core avoids
    // races when the index tensor contains duplicates.
    ctx->SetBlockDim(1);
    ctx->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx)
{
    *ctx->GetOutputShape(0) = *ctx->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static graphStatus InferType(gert::InferDataTypeContext *ctx)
{
    ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char *n) : OpDef(n)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(IndexAdd);
}  // namespace ops
