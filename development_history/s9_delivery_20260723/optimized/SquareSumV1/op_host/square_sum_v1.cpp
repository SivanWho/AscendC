#include "square_sum_v1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>

namespace {
uint32_t Code(ge::DataType d)
{
    return d == ge::DT_FLOAT16 ? 0 : d == ge::DT_FLOAT ? 1 : 2;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *ctx)
{
    const auto &s = ctx->GetInputShape(0)->GetStorageShape();
    uint32_t rank = s.GetDimNum();
    auto *a = ctx->GetAttrs();
    if (rank > 8 || !a || !a->GetListInt(0) || !a->GetBool(1)) {
        return ge::GRAPH_FAILED;
    }
    const auto *axes = a->GetListInt(0);
    uint8_t reduced[8] = {};
    uint32_t uniqueAxisCount = 0;
    if (axes->GetSize() == 0) {
        for (uint32_t d = 0; d < rank; ++d) {
            reduced[d] = 1;
        }
        uniqueAxisCount = rank;
    }
    if (rank == 0) {
        if (axes->GetSize() > 1 ||
            (axes->GetSize() == 1 &&
             axes->GetData()[0] != 0 && axes->GetData()[0] != -1)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        for (size_t i = 0; i < axes->GetSize(); ++i) {
            int64_t d = axes->GetData()[i];
            if (d < 0) {
                d += rank;
            }
            if (d < 0 || d >= rank) {
                return ge::GRAPH_FAILED;
            }
            if (!reduced[d]) {
                reduced[d] = 1;
                ++uniqueAxisCount;
            }
        }
    }
    uint64_t shape[8] = {}, stride[8] = {}, inStride = 1, outCount = 1, reduceCount = 1;
    for (int32_t d = static_cast<int32_t>(rank) - 1; d >= 0; --d) {
        shape[d] = s.GetDim(d);
        stride[d] = inStride;
        inStride *= shape[d];
        if (reduced[d]) {
            reduceCount *= shape[d];
        } else {
            outCount *= shape[d];
        }
    }
    bool seenReduced = false;
    bool suffixReduction = uniqueAxisCount != 0;
    bool contiguousReduction = uniqueAxisCount != 0;
    bool reductionEnded = false;
    for (uint32_t d = 0; d < rank; ++d) {
        if (reduced[d]) {
            if (reductionEnded) {
                contiguousReduction = false;
            }
            seenReduced = true;
        } else if (seenReduced) {
            suffixReduction = false;
            reductionEnded = true;
        }
    }

    const uint32_t dtypeCode = Code(ctx->GetInputDesc(0)->GetDataType());
    const uint64_t lastDim = rank == 0 ? 1 : shape[rank - 1];
    const uint64_t paddedLastDim = (lastDim + 15) / 16 * 16;
    const bool lastAxisOnly =
        rank != 0 && uniqueAxisCount == 1 && reduced[rank - 1] == 1;
    uint32_t fastPath = dtypeCode == 0 && lastAxisOnly && lastDim != 0 &&
                        outCount != 0 &&
                        paddedLastDim <= 64 && outCount <= 255 &&
                        outCount * paddedLastDim <= 32768;
    if (fastPath == 0 && suffixReduction) {
        fastPath = 2;
    } else if (fastPath == 0 && contiguousReduction) {
        fastPath = 3;
    }

    uint32_t blocks = 1;
    SquareSumV1TilingData t;
    t.set_dtypeCode(dtypeCode);
    t.set_rank(rank);
    t.set_axisCount(uniqueAxisCount);
    t.set_blockDim(blocks);
    t.set_fastPath(fastPath);
    t.set_outputCount(outCount);
    t.set_reduceCount(reduceCount);
    t.set_inputShape(shape);
    t.set_inputStride(stride);
    t.set_isReduced(reduced);
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
    auto *a = ctx->GetAttrs();
    if (!a || !a->GetListInt(0) || !a->GetBool(1)) {
        return GRAPH_FAILED;
    }
    bool keep = *a->GetBool(1);
    const size_t rank = x->GetDimNum();
    if (rank > 8) {
        return GRAPH_FAILED;
    }
    bool reduced[8] = {};
    const auto *axes = a->GetListInt(0);
    if (axes->GetSize() == 0) {
        for (size_t d = 0; d < rank; ++d) {
            reduced[d] = true;
        }
    }
    if (rank == 0) {
        if (axes->GetSize() > 1 ||
            (axes->GetSize() == 1 &&
             axes->GetData()[0] != 0 && axes->GetData()[0] != -1)) {
            return GRAPH_FAILED;
        }
    } else {
        for (size_t i = 0; i < axes->GetSize(); ++i) {
            int64_t d = axes->GetData()[i];
            if (d < 0) {
                d += rank;
            }
            if (d < 0 || d >= static_cast<int64_t>(rank)) {
                return GRAPH_FAILED;
            }
            reduced[d] = true;
        }
    }
    auto *y = ctx->GetOutputShape(0);
    size_t outRank = 0;
    for (size_t d = 0; d < rank; ++d) {
        outRank += keep || !reduced[d];
    }
    y->SetDimNum(outRank);
    size_t o = 0;
    for (size_t d = 0; d < rank; ++d) {
        if (reduced[d]) {
            if (keep) {
                y->SetDim(o++, 1);
            }
        } else {
            y->SetDim(o++, x->GetDim(d));
        }
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
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char *n) : OpDef(n)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(REQUIRED).ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}  // namespace ops
