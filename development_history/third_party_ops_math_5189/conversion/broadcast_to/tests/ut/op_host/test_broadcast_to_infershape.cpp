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
 * \file test_broadcast_to_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class BroadcastToInferShapeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "broadcast_to Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "broadcast_to Proto Test TearDown" << std::endl;
  }
};

TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_test_case_1){
    gert::StorageShape shape = {{1,7,2}, {1,7,2}};
    gert::StorageShape shape1 = {{3}, {3}};
    int64_t shapes[4] = {6, 7, 2};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "BroadcastTo",
        { x, shapeParams },
        { y });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{6,7,2}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

