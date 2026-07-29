/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../op_host/histogram_v2_tiling.h"

using namespace ge;
using namespace std;
class HistogramV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HistogramV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HistogramV2Tiling TearDown" << std::endl;
    }
};

struct HistogramV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatform = 0;
    int64_t sysWorkspaceSize = 0;
    NpuArch npuArch = NpuArch::DAV_2002;
};

TEST_F(HistogramV2Tiling, ascend910B1_test_tiling__001)
{
    HistogramV2CompileInfo compileInfo = {64, 262144, 16 * 1024 * 1024};
    gert::TilingContextPara tilingContextPara(
        "HistogramV2",
        {
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{10, 10}, {10, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("bins", Ops::Math::AnyValue::CreateFrom<int64_t>(100)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "100 57344 1 1 1 1 50 2 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    // ExecuteTestCase(tilingContextPara, 0, expectTilingKey, expectTilingData, expectWorkspaces);
}
