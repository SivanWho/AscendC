/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/top_k_v2_tiling_arch35.h"

using namespace std;
using namespace ge;

class TopKV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AscendTopKV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AscendTopKV2Test TearDown" << std::endl;
  }
};


TEST_F(TopKV2Tiling, test_tiling_small_merge_sort) {
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {8};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 8}, {10, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 8}, {10, 8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 13003;
    string expectTilingData = "4294967297 42949672961 4294967306 274877906976 0 4294967297 0 0 1 32 8 1 8 1 0 0 0 0 0 0 0 0 0 0 8589934592 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16787584};
    ExecuteTestCase(
        tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}