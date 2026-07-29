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
 * \file test_transpose_tiling.cpp
 * \brief broadcast_to tiling ut test
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/transpose_tiling_with_gather_arch35.h"

using namespace ge;
// using namespace ut_util;

class TransposeTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TransposeTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TransposeTiling TearDown" << std::endl;
    }
};

TEST_F(TransposeTiling, transpose_tiling_01)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{31, 36, 203, 26}, {31, 36, 203, 26}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10006;
    string expectTilingData =
        "10006 93802085788320 215891311432040488 16842752 27 27 190008 190008 755914244272 0 0 0 190008 0 0 1080 1 1 1 "
        "1 1 1 1 5456 1 1 1 1 1 1 1 176 1 1 1 1 1 1 1 133143986352 4294967297 4294967297 755914244127 4294967297 "
        "4294967297 16557351567361 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
