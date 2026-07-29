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
 * \file test_split_v_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/split_v_tiling_arch35.h"

using namespace std;
using namespace ge;

class SplitVTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitVTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitVTiling TearDown" << std::endl;
  }
};

TEST_F(SplitVTiling, SplitV_test_tiling_001)
{
    optiling::SplitVCompileInfo compileInfo = {32, 0, 253952};
    int32_t size_splits = 1820;
    int32_t split_dim = 0;
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{1820, 232}, {1820, 232}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &size_splits},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
        },
        {
            {{{1820, 232}, {1820, 232}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1))
        },
         &compileInfo);
    uint64_t expectTilingKey = 104;
    string expectTilingData = "253952 0 0 1 1 1 1 0 0 1 0 13195 13216 13195 13216 32 32 32 0 0 422240 422240 0"
                              " -1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}