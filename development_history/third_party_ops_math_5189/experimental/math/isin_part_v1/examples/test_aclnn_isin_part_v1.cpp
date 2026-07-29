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
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_cat.h"
#include "aclnnop/aclnn_sort.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_isin_part_v1.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// 获取Shape对应的元素总数
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
        shapeSize *= i;
    return shapeSize;
}

// 仅创建设备内存和Tensor句柄，不拷贝数据
int CreateEmptyTensor(const std::vector<int64_t>& shape, aclDataType dataType, void** deviceAddr, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 创建aclTensor
template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 结果输出
void PrintOutResult(std::vector<int64_t>& shape, int32_t elementsNum, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("[Output] Result Vector: [");
    for (int64_t i = 0; i < elementsNum; i++) {
        // 使用 %u 打印无符号整数
        LOG_PRINT("%u%s", resultData[i], (i == elementsNum - 1) ? "" : ", ");
    }
    LOG_PRINT("]\n");
}

int main()
{
    // 1. 环境初始化
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateContext(&context, deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return -1);

    // 2. 原始输入数据
    std::vector<float> h_elements = {1.0, 1.0, 4.0, 1.0, 1.0};
    std::vector<float> h_test_elements = {0.0, 1.0, 2.0, 3.0};
    int64_t elementsNum = h_elements.size();
    int64_t totalSize = h_elements.size() + h_test_elements.size();

    std::vector<int64_t> totalShape = {totalSize};
    std::vector<int64_t> outShape = {elementsNum};

    // 3. 资源准备
    void *elAddr = nullptr, *teAddr = nullptr, *catAddr = nullptr;
    void *sortVAddr = nullptr, *sortI64Addr = nullptr, *sortI32Addr = nullptr, *outAddr = nullptr;
    aclTensor *elTensor = nullptr, *teTensor = nullptr, *catTensor = nullptr;
    aclTensor *sortVTensor = nullptr, *sortI64Tensor = nullptr, *sortI32Tensor = nullptr, *outTensor = nullptr;

    CHECK_RET(CreateAclTensor(h_elements, {elementsNum}, &elAddr, ACL_FLOAT, &elTensor) == 0, return -1);
    CHECK_RET(
        CreateAclTensor(h_test_elements, {(int64_t)h_test_elements.size()}, &teAddr, ACL_FLOAT, &teTensor) == 0,
        return -1);

    // 创建输入 Tensor
    CHECK_RET(CreateEmptyTensor(totalShape, ACL_FLOAT, &catAddr, &catTensor) == 0, return -1);
    CHECK_RET(CreateEmptyTensor(totalShape, ACL_FLOAT, &sortVAddr, &sortVTensor) == 0, return -1);
    CHECK_RET(CreateEmptyTensor(totalShape, ACL_INT64, &sortI64Addr, &sortI64Tensor) == 0, return -1);
    CHECK_RET(CreateEmptyTensor(totalShape, ACL_INT32, &sortI32Addr, &sortI32Tensor) == 0, return -1);
    CHECK_RET(CreateEmptyTensor(outShape, ACL_BOOL, &outAddr, &outTensor) == 0, return -1);

    // 4. 调用 aclnnCat
    LOG_PRINT("[Process] Step A: aclnnCat\n");
    std::vector<aclTensor*> catListVec = {elTensor, teTensor};
    aclTensorList* catList = aclCreateTensorList(catListVec.data(), catListVec.size());
    uint64_t wsCat = 0;
    aclOpExecutor* execCat = nullptr;
    CHECK_RET(aclnnCatGetWorkspaceSize(catList, 0, catTensor, &wsCat, &execCat) == ACL_SUCCESS, return -1);
    void* wsCatAddr = nullptr;
    if (wsCat > 0)
        aclrtMalloc(&wsCatAddr, wsCat, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclnnCat(wsCatAddr, wsCat, execCat, stream) == ACL_SUCCESS, return -1);

    // 5. 调用 aclnnSort (输出为 INT64)
    LOG_PRINT("[Process] Step B: aclnnSort (INT64 Indices)\n");
    uint64_t wsSort = 0;
    aclOpExecutor* execSort = nullptr;
    CHECK_RET(
        aclnnSortGetWorkspaceSize(catTensor, false, 0, false, sortVTensor, sortI64Tensor, &wsSort, &execSort) ==
            ACL_SUCCESS,
        return -1);
    void* wsSortAddr = nullptr;
    if (wsSort > 0)
        aclrtMalloc(&wsSortAddr, wsSort, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclnnSort(wsSortAddr, wsSort, execSort, stream) == ACL_SUCCESS, return -1);

    // 6. 调用 aclnnCast (INT64 -> INT32)
    LOG_PRINT("[Process] Step C: aclnnCast (INT64 -> INT32)\n");
    uint64_t wsCast = 0;
    aclOpExecutor* execCast = nullptr;
    CHECK_RET(
        aclnnCastGetWorkspaceSize(sortI64Tensor, ACL_INT32, sortI32Tensor, &wsCast, &execCast) == ACL_SUCCESS,
        return -1);
    void* wsCastAddr = nullptr;
    if (wsCast > 0)
        aclrtMalloc(&wsCastAddr, wsCast, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclnnCast(wsCastAddr, wsCast, execCast, stream) == ACL_SUCCESS, return -1);

    // 7. 调用自定义算子 aclnnIsinPartV1
    LOG_PRINT("[Process] Step D: aclnnIsinPartV1\n");
    uint64_t wsIsin = 0;
    aclOpExecutor* execIsin = nullptr;
    CHECK_RET(
        aclnnIsinPartV1GetWorkspaceSize(sortVTensor, sortI32Tensor, elementsNum, outTensor, &wsIsin, &execIsin) ==
            ACL_SUCCESS,
        return -1);
    void* wsIsinAddr = nullptr;
    if (wsIsin > 0)
        aclrtMalloc(&wsIsinAddr, wsIsin, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclnnIsinPartV1(wsIsinAddr, wsIsin, execIsin, stream) == ACL_SUCCESS, return -1);

    // 8. 同步并获取结果
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);
    std::vector<uint8_t> h_out(elementsNum, 0); // Bool 在 Host 用 uint8_t 接收
    aclrtMemcpy(h_out.data(), elementsNum, outAddr, elementsNum, ACL_MEMCPY_DEVICE_TO_HOST);

    LOG_PRINT("--- Final Result ---\n");
    PrintOutResult(outShape, elementsNum, &outAddr);

    // 9. 资源清理
    aclDestroyTensorList(catList);
    aclDestroyTensor(elTensor);
    aclDestroyTensor(teTensor);
    aclDestroyTensor(catTensor);
    aclDestroyTensor(sortVTensor);
    aclDestroyTensor(sortI64Tensor);
    aclDestroyTensor(sortI32Tensor);
    aclDestroyTensor(outTensor);

    aclrtFree(elAddr);
    aclrtFree(teAddr);
    aclrtFree(catAddr);
    aclrtFree(sortVAddr);
    aclrtFree(sortI64Addr);
    aclrtFree(sortI32Addr);
    aclrtFree(outAddr);

    if (wsCatAddr)
        aclrtFree(wsCatAddr);
    if (wsSortAddr)
        aclrtFree(wsSortAddr);
    if (wsCastAddr)
        aclrtFree(wsCastAddr);
    if (wsIsinAddr)
        aclrtFree(wsIsinAddr);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclFinalize();

    return 0;
}