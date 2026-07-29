# Sign

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：对输入逐元素取符号，正数为1，负数为−1，0为0。
- 计算公式：

  $$
  \text{out}_{i}= 
  \begin{cases}
  1  & \text{if } \text{input}_{i}>0 \\[4pt]
  0  & \text{if } \text{input}_{i}=0 \\[4pt]
  -1 & \text{if } \text{input}_{i}<0
  \end{cases}
  $$

 **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1330px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 230px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>input</td>
      <td>输入</td>
      <td>待取符号的Tensor。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>与input同形状的符号结果。</td>
      <td>数据类型、shape需与input一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>同input</td>
      <td>ND</td>
      <td>√</td>
    </tr>
   </tbody>
  </table>

## 算子使用

使用该算子前，请参考[社区版CANN开发套件包安装文档](../../../docs/zh/invocation/quick_op_invocation.md)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-math
    ```

  - 执行编译

    ```bash
    bash build.sh --pkg --experimental --soc=ascend910b --ops=sign
    ```

  - 部署算子包

    ```bash
    ./build_out/cann-ops-<vendor_name>-linux.<arch>.run
    ```

### 算子调用

  - 执行调用

    ```bash
    bash build.sh --run_example sign eager cust --vendor_name=custom
    ```    

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sign](./examples/test_aclnn_sign.cpp) | 通过[aclnnSign](./docs/aclnnSign.md)接口方式调用Sign算子。 |    

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| hth810 | 个人开发者 | Sign | 2025/12/12 | Sign算子适配开源仓 |
