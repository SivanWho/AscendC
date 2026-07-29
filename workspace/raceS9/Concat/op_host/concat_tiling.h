#ifndef S9_CONCAT_TILING_H
#define S9_CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t CONCAT_MAX_INPUTS = 128;

BEGIN_TILING_DATA_DEF(ConcatTilingData)
TILING_DATA_FIELD_DEF(uint32_t, inputCount);
TILING_DATA_FIELD_DEF(uint32_t, dtypeCode);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerTask);
TILING_DATA_FIELD_DEF(uint32_t, rowTaskCount);
TILING_DATA_FIELD_DEF(uint64_t, outer);
TILING_DATA_FIELD_DEF(uint64_t, inner);
TILING_DATA_FIELD_DEF(uint64_t, outputAxis);
TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_INPUTS, axisSizes);
TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_INPUTS, axisPrefixes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConcatCustom, ConcatTilingData)
}  // namespace optiling

#endif
