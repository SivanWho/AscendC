#ifndef S9_SQUARE_SUM_V1_TILING_H
#define S9_SQUARE_SUM_V1_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)
TILING_DATA_FIELD_DEF(uint32_t, dtypeCode);
TILING_DATA_FIELD_DEF(uint32_t, rank);
TILING_DATA_FIELD_DEF(uint32_t, axisCount);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint64_t, outputCount);
TILING_DATA_FIELD_DEF(uint64_t, reduceCount);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, inputShape);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, inputStride);
TILING_DATA_FIELD_DEF_ARR(uint8_t, 8, isReduced);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)
}  // namespace optiling
#endif
