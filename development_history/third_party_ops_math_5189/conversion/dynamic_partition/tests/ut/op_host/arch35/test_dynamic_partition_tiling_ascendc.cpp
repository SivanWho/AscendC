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
 * \file test_dynamic_partition_tiling_ascendc.cpp
 * \brief dynamic_partition tiling ut test
 */

#include "../../../../op_host/arch35/dynamic_partition_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class DynamicPartitionTilingTest : public testing::Test {
   protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicPartitionTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicPartitionTilingTest TearDown" << std::endl;
    }
};

TEST_F(DynamicPartitionTilingTest, DynamicPartitionTiling_001)
{
    optiling::DynPart::DynamicPartitionCompileInfo compileInfo = {64, 245760, 256, 32};
    gert::TilingContextPara tilingContextPara("DynamicPartition",
                                              {
                                                {{{2, 3, 20, 1000}, {2, 3, 20, 1000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 3, 20}, {2, 3, 20}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{120, 1000}, {120, 1000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 50000;
    string expectTilingData = "50000 60 34359746368 4295213056 2 2 2 1000 1000 1000 1000 1 1000 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}