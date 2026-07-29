
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
 * \file test_broadcast_to_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/broadcast_to_tiling_arch35.h"

using namespace std;
using namespace ge;

class BroadcastToTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BroadcastToTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BroadcastToTiling TearDown" << std::endl;
  }
};

TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_test_failed_1) {
    optiling::BroadcastToCompileInfo compileInfo = {-2, 245760, 256, 256, 32};
    gert::StorageShape shape = {{1, 1, 5}, {1, 1, 5}};
    gert::StorageShape shape1 = {{3}, {3}};
    int32_t shapes[3] = {1, 1, 5};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo",
        {{ shape, ge::DT_FLOAT16, ge::FORMAT_ND }, { shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{ shape, ge::DT_FLOAT16, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 11001;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_test_success_2) {
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape outshape = {{1, 1, 313, 199}, {1, 1, 313, 199}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {1, 1, 313, 199};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo",
        {{ inshape, ge::DT_UINT8, ge::FORMAT_ND }, { shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{ outshape, ge::DT_UINT8, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_test_failed_3) {
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 313, 198}, {1, 1, 313, 198}};
    gert::StorageShape outshape = {{1, 1, 313, 199}, {1, 1, 313, 199}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {1, 1, 313, 199};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo",
        {{ inshape, ge::DT_UINT8, ge::FORMAT_ND }, { shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{ outshape, ge::DT_UINT8, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 1;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}
