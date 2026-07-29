#include "greater_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>

namespace {
uint32_t DtypeCode(ge::DataType dt)
{
    if (dt == ge::DT_FLOAT16) {
        return 0;
    }
    if (dt == ge::DT_FLOAT) {
        return 1;
    }
    if (dt == ge::DT_INT32) {
        return 2;
    }
    if (dt == ge::DT_INT8) {
        return 3;
    }
    return 4;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *ctx)
{
    const auto &a = ctx->GetInputShape(0)->GetStorageShape();
    const auto &b = ctx->GetInputShape(1)->GetStorageShape();
    const uint32_t rank = std::max(a.GetDimNum(), b.GetDimNum());
    if (rank == 0 || rank > 8) {
        return ge::GRAPH_FAILED;
    }
    uint64_t out[8] = {}, sa[8] = {}, sb[8] = {};
    uint64_t aRaw[8] = {}, bRaw[8] = {};
    uint64_t stride = 1;
    for (int i = static_cast<int>(a.GetDimNum()) - 1; i >= 0; --i) {
        aRaw[i] = stride;
        stride *= a.GetDim(i);
    }
    stride = 1;
    for (int i = static_cast<int>(b.GetDimNum()) - 1; i >= 0; --i) {
        bRaw[i] = stride;
        stride *= b.GetDim(i);
    }
    uint64_t total = 1;
    for (uint32_t d = 0; d < rank; ++d) {
        const int32_t ai = static_cast<int32_t>(d) - static_cast<int32_t>(rank - a.GetDimNum());
        const int32_t bi = static_cast<int32_t>(d) - static_cast<int32_t>(rank - b.GetDimNum());
        const uint64_t ad = ai < 0 ? 1 : a.GetDim(ai);
        const uint64_t bd = bi < 0 ? 1 : b.GetDim(bi);
        if (ad != bd && ad != 1 && bd != 1) {
            return ge::GRAPH_FAILED;
        }
        out[d] = std::max(ad, bd);
        sa[d] = (ai < 0 || ad == 1) ? 0 : aRaw[ai];
        sb[d] = (bi < 0 || bd == 1) ? 0 : bRaw[bi];
        total *= out[d];
    }
    bool sameShape = a.GetDimNum() == b.GetDimNum();
    if (sameShape) {
        for (size_t i = 0; i < a.GetDimNum(); ++i) {
            sameShape = sameShape && a.GetDim(i) == b.GetDim(i);
        }
    }
    const uint32_t dtypeCode = DtypeCode(ctx->GetInputDesc(0)->GetDataType());
    const uint32_t fastPath = dtypeCode == 0 && sameShape && total >= 256 && total % 256 == 0;
    uint32_t blocks = 1;
    if (fastPath) {
        auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
        const uint32_t units = total / 256;
        blocks = std::max<uint32_t>(1, std::min<uint32_t>(platform.GetCoreNum(), units));
        while (units % blocks != 0) {
            --blocks;
        }
    }
    GreaterCustomTilingData t;
    t.set_dtypeCode(dtypeCode);
    t.set_rank(rank);
    t.set_blockDim(blocks);
    t.set_fastPath(fastPath);
    t.set_outputCount(total);
    t.set_outputShape(out);
    t.set_x1Strides(sa);
    t.set_x2Strides(sb);
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
    const auto *a = ctx->GetInputShape(0);
    const auto *b = ctx->GetInputShape(1);
    auto *out = ctx->GetOutputShape(0);
    const size_t rank = std::max(a->GetDimNum(), b->GetDimNum());
    out->SetDimNum(rank);
    for (size_t d = 0; d < rank; ++d) {
        const int ai = static_cast<int>(d) - static_cast<int>(rank - a->GetDimNum());
        const int bi = static_cast<int>(d) - static_cast<int>(rank - b->GetDimNum());
        out->SetDim(d, std::max<int64_t>(ai < 0 ? 1 : a->GetDim(ai), bi < 0 ? 1 : b->GetDim(bi)));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferType(gert::InferDataTypeContext *ctx)
{
    ctx->SetOutputDataType(0, ge::DT_BOOL);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class GreaterCustom : public OpDef {
public:
    explicit GreaterCustom(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> in = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32,
                                                        ge::DT_INT8};
        const std::initializer_list<ge::Format> fmt = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                       ge::FORMAT_ND};
        this->Input("self").ParamType(REQUIRED).DataType(in).Format(fmt);
        this->Input("other").ParamType(REQUIRED).DataType(in).Format(fmt);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format(fmt);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(GreaterCustom);
}  // namespace ops
