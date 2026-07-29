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
 * \file test_sort_with_index_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/sort_with_index_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class SortWithIndexTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SortWithIndexTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SortWithIndexTilingTest TearDown" << std::endl;
    }
};

TEST_F(SortWithIndexTilingTest, test_tiling_int32) {
    optiling::SortWithIndexCompileInfo compileInfo;
    compileInfo.core_num = 64;

    gert::TilingContextPara tilingContextPara("SortWithIndex",
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 1003;
    string expectTilingData = "4294967297 2 2 4294967297 2 4294967296 4294967298 2 0 0 0 4294967296 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
