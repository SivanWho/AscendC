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
 * \file test_reflection_pad3d_grad_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../op_host/reflection_pad3d_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class ReflectionPad3dGradTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReflectionPad3dGradTilingTest  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ReflectionPad3dGradTilingTest  TearDown" << std::endl;
    }
};

TEST_F(ReflectionPad3dGradTilingTest, ReflectionPad3dGradTilingData_test_float32_success_case0) {
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{2, 128, 18, 66, 66}, {2, 128, 18, 66, 66}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()},},
                                              {{{{2, 128, 16, 64, 64}, {2, 128, 16, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "549755813890 283467841554 66 343597383760 274877906960 64 0 4294967297 4294967297 4294967297 38482906972208 68719476741 100 ";
    std::vector<size_t> expectWorkspaces = {17268736};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ReflectionPad3dGradTilingTest, ReflectionPad3dGradTilingData_test_float32_success_case1) {
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{2, 64, 18, 130, 130}, {2, 64, 18, 130, 130}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()},},
                                              {{{{2, 64, 16, 128, 128}, {2, 64, 16, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "274877906946 558345748498 130 618475290768 549755813904 128 0 4294967297 4294967297 4294967297 38482906972208 137438953474 101 ";
    std::vector<size_t> expectWorkspaces = {17661952};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ReflectionPad3dGradTilingTest, ReflectionPad3dGradTilingData_test_float16_success_case0) {
    optiling::Tiling4PadV3GradV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> paddingValues = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("ReflectionPad3dGrad",
                                              {{{{2, 128, 18, 22, 1220}, {2, 128, 18, 22, 1220}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingValues.data()},},
                                              {{{{2, 128, 16, 20, 1218}, {2, 128, 16, 20, 1218}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 202;
    string expectTilingData = "549755813890 94489280530 1220 5291399708704 85899345936 1218 0 4294967297 4294967297 4294967297 47828755808304 68719476741 202 ";
    std::vector<size_t> expectWorkspaces = {24346624};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
