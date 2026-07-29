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
 * \file test_aclnn_trace.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "opdev/op_log.h"
#include "level2/aclnn_trace.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_trace_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Trace Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "Trace Test TearDown" << std::endl;
    }
};

// 空tensor
TEST_F(l2_trace_test, case_1)
{
    auto self_tensor_desc = TensorDesc({0, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTrace, INPUT(self_tensor_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

// CheckNotNull self
TEST_F(l2_trace_test, case_2)
{
    auto tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTrace, INPUT(nullptr), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(l2_trace_test, case_3)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTrace, INPUT(tensor_desc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_trace_test, case_4)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_BOOL, ValidList[i]);
        auto out_desc = TensorDesc({}, ACL_INT64, ValidList[i]).Precision(0.001, 0.001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// CheckShape
TEST_F(l2_trace_test, case_5)
{
    auto self_desc = TensorDesc({4, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// float
TEST_F(l2_trace_test, case_6)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_FLOAT, ValidList[i]).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_FLOAT, ValidList[i]).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// not contiguous
TEST_F(l2_trace_test, case_7)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({2, 4}, ACL_FLOAT, ValidList[i], {1, 2}, 0, {4, 2}).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_FLOAT, ValidList[i]).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

TEST_F(l2_trace_test, case_8)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_FLOAT16, ValidList[i]).ValueRange(-100, 100);
        auto out_desc = TensorDesc({}, ACL_FLOAT16, ValidList[i]).Precision(0.001, 0.001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// not contiguous
TEST_F(l2_trace_test, case_9)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({2, 4}, ACL_FLOAT16, ValidList[i], {1, 2}, 0, {4, 2}).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_FLOAT16, ValidList[i]).Precision(0.001, 0.001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// CheckShape
TEST_F(l2_trace_test, case_10)
{
    auto self_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// INT64
TEST_F(l2_trace_test, case_11)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_INT64, ValidList[i]).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_INT64, ValidList[i]).Precision(0.001, 0.001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// COMPLEX64
TEST_F(l2_trace_test, case_12)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_COMPLEX64, ValidList[i]).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_COMPLEX64, ValidList[i]).Precision(0.001, 0.001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// COMPLEX128
TEST_F(l2_trace_test, case_13)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_COMPLEX128, ValidList[i]).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_COMPLEX128, ValidList[i]).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// INT32
TEST_F(l2_trace_test, case_14)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_INT32, ValidList[i]).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_INT64, ValidList[i]).Precision(0.001, 0.001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// UINT8
TEST_F(l2_trace_test, case_15)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_UINT8, ValidList[i]).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_INT64, ValidList[i]).Precision(0.001, 0.001);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        ut.TestPrecision();
    }
}

// BFLOAT16
TEST_F(l2_trace_test, Ascend910B2_case_16)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_NCHW, ACL_FORMAT_NHWC,  ACL_FORMAT_ND,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({4, 4}, ACL_BF16, ValidList[i]).ValueRange(-1000, 1000);
        auto out_desc = TensorDesc({}, ACL_BF16, ValidList[i]).Precision(0.004, 0.004);
        auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}