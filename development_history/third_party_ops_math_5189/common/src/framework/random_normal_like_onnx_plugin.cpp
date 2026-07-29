/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "op_math_proto_extend.h"
#include "math/mul/op_graph/mul_proto.h"
#include "math/add/op_graph/add_proto.h"
#include "math/cast/op_graph/cast_proto.h"
#include "math/shape/op_graph/shape_proto.h"

using namespace ge;
namespace domi {
static Status ParseParamsRandomNormalLike(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    op_dest.SetAttr("original_type", "ai.onnx::11::RandomNormalLike");

    int dtype = 1;
    float mean = 0.0f;
    float scale = 1.0f;
    int seed = 0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "dtype") {
            dtype = attr.i();
        } else if (attr.name() == "mean") {
            mean = attr.f();
        } else if (attr.name() == "scale") {
            scale = attr.f();
        } else if (attr.name() == "seed") {
            seed = (int)attr.f();
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("mean", mean);
    op_dest.SetAttr("scale", scale);
    op_dest.SetAttr("seed", seed);
    op_dest.SetAttr("dtype", dtype);
    return SUCCESS;
}

static Status ParseOpToGraphRandomNormalLike(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    float mean = 0.0f;
    op.GetAttr("mean", mean);

    float scale = 1.0f;
    op.GetAttr("scale", scale);

    int dtype = 1;
    op.GetAttr("dtype", dtype);

    // cast from onnx dtype to tbe dtype
    std::map<int, ge::DataType> kvlist = {{1, ge::DT_FLOAT}, {10, ge::DT_FLOAT16}, {11, ge::DT_DOUBLE}};
    if (kvlist.find(dtype) == kvlist.end()) {
        OP_LOGE(GetOpName(op).c_str(), "only support float32/float16/double, but got %d", dtype);
        return FAILED;
    }

    int seed = 0;
    op.GetAttr("seed", seed);

    ge::Tensor scalar_mean = CreateScalar(mean, ge::DT_FLOAT);
    ge::Tensor scalar_scale = CreateScalar(scale, ge::DT_FLOAT);

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto shape_op = op::Shape((ori_name + "_shape").c_str()).set_input_x(data0).set_attr_dtype(ge::DT_INT32);
    auto random_op = op::RandomStandardNormal((ori_name + "_random_standard_normal").c_str())
                         .set_input_shape(shape_op)
                         .set_attr_dtype(kvlist[dtype])
                         .set_attr_seed(seed)
                         .set_attr_seed2(seed);
    auto const_scale = op::Const((ori_name + "_const_scale").c_str()).set_attr_value(scalar_scale);
    auto const_mean = op::Const((ori_name + "_const_mean").c_str()).set_attr_value(scalar_mean);
    auto cast_mean =
        op::Cast((ori_name + "_cast_mean").c_str()).set_input_x(const_mean).set_attr_dst_type(kvlist[dtype]);
    auto cast_scale =
        op::Cast((ori_name + "_cast_scale").c_str()).set_input_x(const_scale).set_attr_dst_type(kvlist[dtype]);

    auto mul_op = op::Mul((ori_name + "_mul").c_str()).set_input_x1(random_op).set_input_x2(cast_scale);
    auto add_op = op::Add((ori_name + "_add").c_str()).set_input_x1(mul_op).set_input_x2(cast_mean);
    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(add_op, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::RandomNormalLike"), ge::AscendString("ai.onnx::9::RandomNormalLike"),
         ge::AscendString("ai.onnx::10::RandomNormalLike"), ge::AscendString("ai.onnx::11::RandomNormalLike"),
         ge::AscendString("ai.onnx::12::RandomNormalLike"), ge::AscendString("ai.onnx::13::RandomNormalLike"),
         ge::AscendString("ai.onnx::14::RandomNormalLike"), ge::AscendString("ai.onnx::15::RandomNormalLike"),
         ge::AscendString("ai.onnx::16::RandomNormalLike"), ge::AscendString("ai.onnx::17::RandomNormalLike"),
         ge::AscendString("ai.onnx::18::RandomNormalLike")})
    .ParseParamsFn(ParseParamsRandomNormalLike)
    .ParseOpToGraphFn(ParseOpToGraphRandomNormalLike)
    .ImplyType(ImplyType::TVM);
} // namespace domi
