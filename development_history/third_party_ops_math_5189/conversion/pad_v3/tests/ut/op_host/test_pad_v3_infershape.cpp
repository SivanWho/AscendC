/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PadV3InfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PadV3InfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PadV3InfershapeTest TearDown" << std::endl;
    }
};

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_0)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_1_different_paddings)
{
    gert::StorageShape xShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape padShape = {{2, 4}, {2, 4}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_2_int64_dtype)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_3_wrong_paddings_num)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    std::vector<int32_t> padValues = {1, 2, 3};
    gert::StorageShape padShape = {{3}, {3}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_4_negative_output)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    std::vector<int32_t> padValues = {-5, -6, -5, -6, -5, -6};
    gert::StorageShape padShape = {{6}, {6}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_5_unknown_rank)
{
    gert::StorageShape xShape = {{-2}, {-2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_6_2d_input)
{
    gert::StorageShape xShape = {{3, 5}, {3, 5}};
    gert::StorageShape padShape = {{2}, {2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_7_1d_input)
{
    gert::StorageShape xShape = {{10}, {10}};
    gert::StorageShape padShape = {{2}, {2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_8_4d_input)
{
    gert::StorageShape xShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape padShape = {{4}, {4}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_10_empty_input)
{
    gert::StorageShape xShape = {{}, {{}}};
    gert::StorageShape padShape = {{2}, {2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_11_with_values_attr)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    std::vector<int64_t> paddingsValues = {1, 2, 3, 4, 5, 6};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, false, nullptr},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr(
            "paddings_values", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_12_unknown_dim_in_x)
{
    gert::StorageShape xShape = {{-1, 3, 4}, {-1, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_13_zero_padding)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{3}, {3}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_14_large_padding)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{10, 10}, {10, 10}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}