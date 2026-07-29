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
 * \file test_transpose_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;
class TransposeInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TransposeInferShapeTest Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TransposeInferShapeTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(TransposeInferShapeTest, transpose_test_case_1)
{
    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Transpose",
        {x, perm},
        {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransposeInferShapeTest, transpose_test_case_2)
{
    int32_t perm_value[4] = {3, 0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Transpose",
        {x, perm},
        {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransposeInferShapeTest, transpose_test_case_3)
{
    int16_t perm_value[4] = {3, 0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT16, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Transpose",
        {x, perm},
        {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(TransposeInferShapeTest, transpose_test_case_4)
{
    int16_t perm_value[4] = {3, 0, 4, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT16, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Transpose",
        {x, perm},
        {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}