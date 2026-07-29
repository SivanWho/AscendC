#ifndef S9_INDEX_ADD_TILING_H
#define S9_INDEX_ADD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
TILING_DATA_FIELD_DEF(uint32_t, dtypeCode);
TILING_DATA_FIELD_DEF(uint32_t, rank);
TILING_DATA_FIELD_DEF(uint64_t, selfCount);
TILING_DATA_FIELD_DEF(uint64_t, sourceCount);
TILING_DATA_FIELD_DEF(uint64_t, indexCount);
TILING_DATA_FIELD_DEF(uint64_t, selfDim);
TILING_DATA_FIELD_DEF(uint64_t, inner);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}  // namespace optiling
#endif
