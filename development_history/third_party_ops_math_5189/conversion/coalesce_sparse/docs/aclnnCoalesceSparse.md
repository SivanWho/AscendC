# aclnnCoalesceSparse

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/coalesce_sparse)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

将相同坐标点的value进行累加求和，进而减少Coo_Tensor的内存大小。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCoalesceSparseGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCoalesceSparse”接口执行计算。

```Cpp
aclnnStatus aclnnCoalesceSparseGetWorkspaceSize(
    const aclTensor   *uniqueLen,
    const aclTensor   *uniqueIndices,
    const aclTensor   *indices,
    const aclTensor   *values,
    const aclTensor   *newIndicesOut,
    const aclTensor   *newValuesOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```Cpp
aclnnStatus aclnnCoalesceSparse(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream);
```

## aclnnCoalesceSparseGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1519px"><colgroup>
  <col style="width: 217px">
  <col style="width: 120px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>uniqueLen（aclTensor*）</td>
      <td>输入</td>
      <td>去重后的索引数。</td>
      <td>不支持空Tensor。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>uniqueIndices（aclTensor*）</td>
      <td>输入</td>
      <td>去重后的索引数组。</td>
      <td>不支持空Tensor。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices（aclTensor*）</td>
      <td>输入</td>
      <td>索引数组。</td>
      <td><ul><li>不支持空Tensor。</li><li>重索引后的indices值不能超过int32上限。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>values（aclTensor*）</td>
      <td>输入</td>
      <td>每个坐标对应的元素值。</td>
      <td>不支持空Tensor。</td>
      <td>INT32、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>newIndicesOut（aclTensor*）</td>
      <td>输出</td>
      <td>合并后的索引数组。</td>
      <td>不支持空Tensor。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>newValuesOut（aclTensor*）</td>
      <td>输出</td>
      <td>合并后的元素值。</td>
      <td>不支持空Tensor。</td>
      <td>INT32、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的uniqueLen、uniqueIndices、indices、values、newIndicesOut或newValuesOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>uniqueLen、uniqueIndices、indices、values、newIndicesOut或newValuesOut的数据类型不在支持范围之内。</td>
    </tr>
    <tr>
      <td>values或newValuesOut的维度超过8维。</td>
    </tr>
    <tr>
      <td>重索引后的indices值不能超过int32上限。</td>
    </tr>
  </tbody></table>

## aclnnCoalesceSparse

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCoalesceSparseGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

只能从PTA侧调用，如想使用aclnn接口需要手动构造unique_indices入参，并确保调用Kernel前new_values Tensor已置为全零。

重索引后的indices值不能超过int32上限。

确定性计算：
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas 推理系列产品</term>、 <term>Atlas 训练系列产品</term>：aclnnCoalesceSparse默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。
- <term>Ascend 950PR/Ascend 950DT</term>：aclnnCoalesceSparse默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_coalesce_sparse.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API文档
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> uniqueLenShape = {1};
    std::vector<int64_t> uniqueIndicesShape = {2,2};
    std::vector<int64_t> indexShape = {2,4};
    std::vector<int64_t> valueShape = {4};
    std::vector<int64_t> newIndexShape = {4};
    std::vector<int64_t> newValueShape = {2};
    void* uniqueLenDeviceAddr = nullptr;
    void* uniqueIndicesDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* newIndexDeviceAddr = nullptr;
    void* newValueDeviceAddr = nullptr;
    aclTensor* uniqueLen = nullptr;
    aclTensor* uniqueIndices = nullptr;
    aclTensor* index = nullptr;
    aclTensor* value = nullptr;
    aclTensor* newIndex = nullptr;
    aclTensor* newValue = nullptr;
    std::vector<int32_t> uniqueLenData = {2};
    std::vector<int32_t> uniqueIndicesData = {0, 1, 0, 2};
    std::vector<int32_t> indexData = {0, 0, 1, 1, 0, 0, 2, 2};
    std::vector<float> valueData = {1, 2, 3, 4};
    std::vector<int32_t> newIndexData = {0, 0, 0, 0};
    std::vector<float> newValueData = {0, 0};

    // 创建in aclTensor
    ret = CreateAclTensor(uniqueLenData, uniqueLenShape, &uniqueLenDeviceAddr, aclDataType::ACL_INT32, &uniqueLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(uniqueIndicesData, uniqueIndicesShape, &uniqueIndicesDeviceAddr, aclDataType::ACL_INT32, &uniqueIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(indexData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(valueData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(newIndexData, newIndexShape, &newIndexDeviceAddr, aclDataType::ACL_INT32, &newIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(newValueData, newValueShape, &newValueDeviceAddr, aclDataType::ACL_FLOAT, &newValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnCoalesceSparse第一段接口
    ret = aclnnCoalesceSparseGetWorkspaceSize(uniqueLen, uniqueIndices, index, value, newIndex, newValue, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCoalesceSparseGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnCoalesceSparse第二段接口
    ret = aclnnCoalesceSparse(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCoalesceSparse failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(newValueShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), newValueDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(uniqueLen);
    aclDestroyTensor(uniqueIndices);
    aclDestroyTensor(index);
    aclDestroyTensor(value);
    aclDestroyTensor(newIndex);
    aclDestroyTensor(newValue);

    // 7.释放device资源
    aclrtFree(uniqueLenDeviceAddr);
    aclrtFree(uniqueIndicesDeviceAddr);
    aclrtFree(indexDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(newIndexDeviceAddr);
    aclrtFree(newValueDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
