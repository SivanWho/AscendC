#ifndef S9_GREATER_TILING_H
#define S9_GREATER_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GreaterCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, dtypeCode);
TILING_DATA_FIELD_DEF(uint32_t, rank);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint64_t, outputCount);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outputShape);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, x1Strides);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, x2Strides);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Greater, GreaterCustomTilingData)
}  // namespace optiling
#endif
