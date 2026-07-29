/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_histc.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_histc_test : public testing::Test {
  protected:
  static void SetUpTestCase() { cout << "histc_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "histc_test TearDown" << endl; }
};

TEST_F(l2_histc_test, case_000_workspace) {
  auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-9.0f);
  auto maxScalar = ScalarDesc(9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_001_float32_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-9.0f);
  auto maxScalar = ScalarDesc(9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_002_float16_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-9.0f);
  auto maxScalar = ScalarDesc(9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_003_int32_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-9.0f);
  auto maxScalar = ScalarDesc(9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_008_1_dim_input_tensor) {
  auto selfTensor = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-9.0f);
  auto maxScalar = ScalarDesc(9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_009_3_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-9.0f);
  auto maxScalar = ScalarDesc(9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_010_5_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-9.0f);
  auto maxScalar = ScalarDesc(9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_011_8_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-9.0f);
  auto maxScalar = ScalarDesc(9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_012_bins_coverage) {
  auto selfTensor = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 10;
  auto minScalar = ScalarDesc(-10.0f);
  auto maxScalar = ScalarDesc(10.0f);

  auto ut1 = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  outTensor = TensorDesc({12}, ACL_FLOAT, ACL_FORMAT_ND);
  bins = 12;
  auto ut2 = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
  workspaceSize = 0;
  aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  outTensor = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
  bins = 14;
  auto ut3 = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
  workspaceSize = 0;
  aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_014_NHWC) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NHWC);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-10.0f);
  auto maxScalar = ScalarDesc(10.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_015_NCHW) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-10.0f);
  auto maxScalar = ScalarDesc(10.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_016_HWCN) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_HWCN);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(-10.0f);
  auto maxScalar = ScalarDesc(10.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_018_empty_tensor) {
  auto selfTensor = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 2;
  auto minScalar = ScalarDesc(-10.0f);
  auto maxScalar = ScalarDesc(10.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_histc_test, case_019_float32_min_greater_max) {
  auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(9.0f);
  auto maxScalar = ScalarDesc(-9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_histc_test, case_020_min_greater_max) {
  auto selfTensor = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
  int64_t bins = 3;
  auto minScalar = ScalarDesc(9.0f);
  auto maxScalar = ScalarDesc(-9.0f);

  auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
