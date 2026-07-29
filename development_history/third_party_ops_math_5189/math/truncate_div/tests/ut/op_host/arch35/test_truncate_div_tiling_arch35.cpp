/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_truncate_div_tiling_arch35.cpp
 * \brief TruncateDiv tiling test
 */

#include "math/truncate_div/op_host/arch35/truncate_div_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace std;
using namespace ge;
using namespace Ops::Base;

class TruncateDivTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TruncateDivTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TruncateDivTilingTest TearDown" << std::endl;
    }
};

TEST_F(TruncateDivTilingTest, truncate_div_fp16_fp_scalar)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, x2.data()},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b1'00000000'00000111;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_fp_fp1)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b0'00000000'00001000;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_bf16_1)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{17772, 1, 2, 1, 2, 1, 2, 1}, {17772, 1, 2, 1, 2, 1, 2, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{17772, 2, 2, 2, 2, 2, 2, 2}, {17772, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b0'00000000'00000001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_f32_scalar_3)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{17772, 1, 2, 1, 2, 1, 2, 1}, {17772, 1, 2, 1, 2, 1, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, x2.data()},
        },
        {
            {{{17772, 2, 2, 2, 2, 2, 2, 2}, {17772, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b1'00000000'00000001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_f16_scalar_4)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{17772, 1, 2, 1, 2, 1, 2, 1}, {17772, 1, 2, 1, 2, 1, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, x2.data()},
        },
        {
            {{{17772, 2, 2, 2, 2, 2, 2, 2}, {17772, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b1'00000000'00000001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, truncate_div_bf16_scalar_5)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{17772, 1, 2, 1, 2, 1, 2, 1}, {17772, 1, 2, 1, 2, 1, 2, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND, true, x2.data()},
        },
        {
            {{{17772, 2, 2, 2, 2, 2, 2, 2}, {17772, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0b1'00000000'00000001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, test_ez0020_scalar_dtype_int32)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    
    std::vector<int32_t> x2 = {4};
    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, x2.data()},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(TruncateDivTilingTest, test_ez0021_dtype_combination_double)
{
    BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;

    gert::TilingContextPara tilingContextPara(
        "TruncateDiv",
        {
            {{{8, 128}, {8, 128}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{8, 128}, {8, 128}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}