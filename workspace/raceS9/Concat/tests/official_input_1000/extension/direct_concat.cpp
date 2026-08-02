// Test-only extension. It invokes the same ACLNN ABI as the official harness,
// but exactly once per generated test case. It is never packaged for submission.
#include <torch/extension.h>

#include "../common/pytorch_npu_helper.hpp"

using namespace at;
using tensor_list = std::vector<at::Tensor>;

at::Tensor concat_once(
    const tensor_list &inputs,
    int64_t dim,
    const at::IntArrayRef &output_shape)
{
    TORCH_CHECK(!inputs.empty(), "Concat test input list must not be empty");
    at::Tensor result = at::empty(output_shape, inputs[0].options());
    at::TensorList input_list(inputs);
    EXEC_NPU_CMD(aclnnConcat, input_list, dim, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("concat_once", &concat_once, "Single ACLNN Concat invocation");
}
