/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file clip_by_value_v2_ub_broadcast_tiling.cpp
 * \brief
 */
#include <vector>
#include "op_host/math_tiling_templates_registry.h"
#include "conversion/clip_by_value/op_host/arch35/clip_by_value_tiling.h"
#include "conversion/clip_by_value/op_host/arch35/clip_by_value_ub_broadcast_tiling.h"

using namespace ge;
namespace optiling {
static constexpr uint64_t CLIP_BY_VALUE_UB_BROADCAST_TILING_PRIORITY = 2;

REGISTER_OPS_TILING_TEMPLATE(ClipByValueV2, ClipByValueTilingUbBroadcast, CLIP_BY_VALUE_UB_BROADCAST_TILING_PRIORITY);

} // namespace optiling