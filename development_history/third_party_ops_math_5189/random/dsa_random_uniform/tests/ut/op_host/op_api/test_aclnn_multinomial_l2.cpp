/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../../../../op_host/op_api/aclnn_multinomial.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_multinomial_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_multinomial_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_multinomial_test TearDown" << endl;
    }
};

TEST_F(l2_multinomial_test, aclnnMultinomial_6000_float_false_2000_nd) {
  const vector<int64_t>& self_shape = {6000};
  aclDataType self_dtype = ACL_FLOAT;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t numsamples = 2000;
  bool replacement = false;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {2000};
  aclDataType out_dtype = ACL_INT64;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(1,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, numsamples, replacement, seed, offset),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multinomial_test, aclnnMultinomial_6000_float_false_1_nd) {
  const vector<int64_t>& self_shape = {6000};
  aclDataType self_dtype = ACL_FLOAT;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t numsamples = 1;
  bool replacement = false;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {1};
  aclDataType out_dtype = ACL_INT64;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(1,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, numsamples, replacement, seed, offset),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multinomial_test, aclnnMultinomial_6000_float16_false_2000_nd) {
  const vector<int64_t>& self_shape = {6000};
  aclDataType self_dtype = ACL_FLOAT16;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t numsamples = 2000;
  bool replacement = false;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {2000};
  aclDataType out_dtype = ACL_INT64;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(1,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, numsamples, replacement, seed, offset),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multinomial_test, aclnnMultinomial_5_4_6_int32_t_nd) {
  const vector<int64_t>& self_shape = {5,4};
  aclDataType self_dtype = ACL_INT32;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t from = 2;
  int64_t to = 4;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {5,4};
  aclDataType out_dtype = ACL_INT32;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(-5,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_multinomial_test, aclnnMultinomial_out_int16_t_nd) {
  const vector<int64_t>& self_shape = {5,4};
  aclDataType self_dtype = ACL_FLOAT16;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t from = 2;
  int64_t to = 4;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {5,4};
  aclDataType out_dtype = ACL_INT16;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(-5,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_multinomial_test, aclnnMultinomial_self_nullptr) {

  int64_t from = 2;
  int64_t to = 4;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {5,4};
  aclDataType out_dtype = ACL_UINT8;
  aclFormat out_format = ACL_FORMAT_NHWC;

  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(nullptr, from, to, seed, offset),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

}

TEST_F(l2_multinomial_test, aclnnMultinomial_out_nullptr) {
  const vector<int64_t>& self_shape = {3,4};
  aclDataType self_dtype = ACL_DOUBLE;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t numsamples = 2;
  bool replacement = false;
  int64_t seed = 0;
  int64_t offset = 0;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(0,5);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, numsamples, replacement, seed, offset),
                      OUTPUT(nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multinomial_test, aclnnMultinomial_5_4_6_hwcn) {
  const vector<int64_t>& self_shape = {5,4,6};
  aclDataType self_dtype = ACL_FLOAT16;
  aclFormat self_format = ACL_FORMAT_HWCN;

  int64_t from = 2;
  int64_t to = 4;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {5,4,6};
  aclDataType out_dtype = ACL_INT64;
  aclFormat out_format = ACL_FORMAT_HWCN;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, from, to, seed, offset),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_multinomial_test, aclnnMultinomial_numsamples_exception_nd) {
  const vector<int64_t>& self_shape = {3,4};
  aclDataType self_dtype = ACL_FLOAT16;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t numsamples = -2;
  bool replacement = false;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {3,4};
  aclDataType out_dtype = ACL_INT64;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(0,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, numsamples, replacement, seed, offset), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_multinomial_test, aclnnMultinomial_exception_nd) {
  const vector<int64_t>& self_shape = {3,4};
  aclDataType self_dtype = ACL_FLOAT16;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t numsamples = 5;
  bool replacement = false;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {3,4};
  aclDataType out_dtype = ACL_INT64;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(0,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, numsamples, replacement, seed, offset), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_multinomial_test, ascend910B2_aclnnMultinomial_6000_bf16_false_2000_nd) {
  const vector<int64_t>& self_shape = {6000};
  aclDataType self_dtype = ACL_BF16;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t numsamples = 2000;
  bool replacement = false;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {2000};
  aclDataType out_dtype = ACL_INT64;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(1,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, numsamples, replacement, seed, offset),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multinomial_test, ascend950PR_89_aclnnMultinomial_6000_float_false_2000_nd) {
  const vector<int64_t>& self_shape = {6000};
  aclDataType self_dtype = ACL_FLOAT;
  aclFormat self_format = ACL_FORMAT_ND;

  int64_t numsamples = 2000;
  bool replacement = false;
  int64_t seed = 0;
  int64_t offset = 0;

  const vector<int64_t>& out_shape = {2000};
  aclDataType out_dtype = ACL_INT64;
  aclFormat out_format = ACL_FORMAT_ND;

  auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format).ValueRange(1,5);
  auto out_tensor_desc = TensorDesc(out_shape, out_dtype, out_format);

  auto ut = OP_API_UT(aclnnMultinomial,
                      INPUT(self_tensor_desc, numsamples, replacement, seed, offset),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}