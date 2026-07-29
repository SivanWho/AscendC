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
 * \file test_clip_by_value_v2_tiling.cpp
 * \brief
 */
#include "conversion/clip_by_value/op_host/arch35/clip_by_value_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
class ClipByValueV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ClipByValueV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ClipByValueV2Tiling TearDown" << std::endl;
    }
};

TEST_F(ClipByValueV2Tiling, clipByValuev2_FP16_Sig_Dim)
{
    optiling::ClipByValueCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ClipByValueV2",
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2001100;
    string expectTilingData = "32 6144 6144 1 1 196608 1 1 196608 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ClipByValueV2Tiling, clipByValuev2_FP16_UB_Broadcast)
{
    optiling::ClipByValueCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ClipByValueV2",
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 3000002;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ClipByValueV2Tiling, clipByValuev2_FP16_NDDMA)
{
    optiling::ClipByValueCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ClipByValueV2",
        {
            {{{48, 16, 16, 15}, {48, 16, 16, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{48, 16, 16, 15}, {48, 16, 16, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100000001000100;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ClipByValueV2Tiling, clipByValuev2_FP32_NDDMA)
{
    optiling::ClipByValueCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ClipByValueV2",
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 200000001000100;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ClipByValueV2Tiling, clipByValuev2_INT32_NDDMA)
{
    optiling::ClipByValueCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ClipByValueV2",
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 400000001000100;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ClipByValueV2Tiling, clipByValuev2_INT64_NDDMA)
{
    optiling::ClipByValueCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ClipByValueV2",
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 16, 16, 1}, {48, 16, 16, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{48, 16, 16, 16}, {48, 16, 16, 16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 500000001000100;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
