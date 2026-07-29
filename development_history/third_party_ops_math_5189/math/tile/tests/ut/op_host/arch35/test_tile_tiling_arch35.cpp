/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_tile_tiling_arch35.cpp
 * \brief tile tiling ut test
 */

#include "../../../../op_host/arch35/tile_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;

class TileTiling : public testing::Test
{
protected:
    static void SetUpTestCase() { std::cout << "TileTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TileTiling TearDown" << std::endl; }
};

TEST_F(TileTiling, TileTiling_001)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;  // 240 * 1024
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    gert::StorageShape xShape = {{2, 3, 1, 1000, 1, 1}, {2, 3, 1, 1000, 1, 1}};
    gert::StorageShape shape = {{2, 3, 1, 1, 2, 2}, {2, 3, 1, 1, 2, 2}};
    gert::StorageShape yShape = {{4, 9, 1, 1000, 2, 2}, {4, 9, 1, 1000, 2, 2}};
    gert::TilingContextPara tilingContextPara(
        "Tile",
        {{ xShape, ge::DT_FLOAT, ge::FORMAT_ND }, { shape, ge::DT_INT32, ge::FORMAT_ND }},
        {{ yShape, ge::DT_FLOAT, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 11000;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}
