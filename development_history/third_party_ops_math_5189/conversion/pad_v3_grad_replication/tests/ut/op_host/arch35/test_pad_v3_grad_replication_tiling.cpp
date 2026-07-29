/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "conversion/pad_v3_grad_replication/op_host/arch35/pad_v3_grad_replication_tiling.h"

using namespace std;
using namespace ge;
using namespace optiling;

class PadV3GradReplicationTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PadV3GradReplicationTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PadV3GradReplicationTilingTest TearDown" << std::endl;
    }
};

// Test 1: 1D case, float32, simple padding
TEST_F(PadV3GradReplicationTilingTest, test_1d_float32_basic)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32,
        .ub_size = 128 * 1024, // 128KB UB
        .sysWorkspaceSize = 16 * 1024 * 1024};

    // Input x (padding后): shape=[70], 含padding (left=10, right=10)
    gert::StorageShape xShape = {{70}, {70}};
    gert::StorageShape paddingsShape = {{2}, {2}};
    std::vector<int32_t> paddingsValue = {10, 10}; // left=10, right=10

    // Output y (原始): shape=[50]
    gert::StorageShape yShape = {{50}, {50}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (1 - 1); // dimNum=1, splitAxis=0 -> 0
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test 2: 2D case, float16, large shape
TEST_F(PadV3GradReplicationTilingTest, test_2d_float16_large)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // Input x (padding后): [522, 64] (H+padding, W)
    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0}; // H维度: left=10, right=0

    // Output y (原始): [512, 64]
    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1); // dimNum=2, splitAxis=0 -> 1
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test 3: 3D case, bfloat16, all dimensions have padding
TEST_F(PadV3GradReplicationTilingTest, test_3d_bfloat16_all_padding)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // Input x (padding后): [15, 522, 74] (B+padding, H+padding, W+padding)
    gert::StorageShape xShape = {{15, 522, 74}, {15, 522, 74}};
    gert::StorageShape paddingsShape = {{6}, {6}};
    std::vector<int32_t> paddingsValue = {5, 0, 10, 0, 5, 5}; // B:5, H:10, W:10

    // Output y (原始): [10, 512, 64]
    gert::StorageShape yShape = {{10, 512, 64}, {10, 512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_BF16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (1ULL << 4) | (3 - 1); // dimNum=3, splitAxis=1 -> 18
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test 4: 4D case, float32, NCHW format
TEST_F(PadV3GradReplicationTilingTest, test_4d_float32_nchw)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // Input x (padding后): [10, 3, 522, 74]
    gert::StorageShape xShape = {{10, 3, 522, 74}, {10, 3, 522, 74}};
    gert::StorageShape paddingsShape = {{8}, {8}};
    std::vector<int32_t> paddingsValue = {0, 0, 0, 0, 10, 0, 5, 5}; // N:0, C:0, H:10, W:10

    // Output y (原始): [10, 3, 512, 64]
    gert::StorageShape yShape = {{10, 3, 512, 64}, {10, 3, 512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (2ULL << 4) | (4 - 1); // dimNum=4, splitAxis=2 -> 35
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test 5: 6D case, float32, only last 5 dims have padding (first dim padding=0)
TEST_F(PadV3GradReplicationTilingTest, test_6d_float32_last5_padding)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // Input x (padding后): [100, 10, 20, 522, 74, 110]
    // First dim (N=100) has no padding
    gert::StorageShape xShape = {{100, 10, 20, 522, 74, 110}, {100, 10, 20, 522, 74, 110}};
    gert::StorageShape paddingsShape = {{12}, {12}};
    std::vector<int32_t> paddingsValue = {0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 5, 5}; // 前6维padding=0

    // Output y (原始): [100, 10, 20, 512, 64, 100]
    gert::StorageShape yShape = {{100, 10, 20, 512, 64, 100}, {100, 10, 20, 512, 64, 100}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (3ULL << 4) | (6 - 1); // dimNum=6, splitAxis=3 -> 53
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test 6: 8D case, float16, only last 5 dims have padding
TEST_F(PadV3GradReplicationTilingTest, test_8d_float16_last5_padding)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // Input x (padding后): [2, 3, 4, 100, 10, 20, 522, 110]
    // First 3 dims have no padding
    gert::StorageShape xShape = {{2, 3, 4, 100, 10, 20, 522, 110}, {2, 3, 4, 100, 10, 20, 522, 110}};
    gert::StorageShape paddingsShape = {{16}, {16}};
    std::vector<int32_t> paddingsValue = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10};

    // Output y (原始): [2, 3, 4, 100, 10, 20, 512, 100]
    gert::StorageShape yShape = {{2, 3, 4, 100, 10, 20, 512, 100}, {2, 3, 4, 100, 10, 20, 512, 100}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (6ULL << 4) | (8 - 1); // dimNum=8, splitAxis=6 -> 103
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test 7: paddings_contiguous=false (non-contiguous padding array)
TEST_F(PadV3GradReplicationTilingTest, test_3d_float32_non_contiguous_padding)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{15, 522, 74}, {15, 522, 74}};
    gert::StorageShape paddingsShape = {{6}, {6}};
    // Non-contiguous: [left0, left1, left2, right0, right1, right2]
    std::vector<int64_t> paddingsValue = {5, 10, 5, 0, 0, 5};

    gert::StorageShape yShape = {{10, 512, 64}, {10, 512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);

    uint64_t expectedTilingKey = (1ULL << 4) | (3 - 1); // 18
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}