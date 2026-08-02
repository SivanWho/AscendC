import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


torch_npu_root = os.path.dirname(os.path.abspath(torch_npu.__file__))

setup(
    name="concat_direct_test_ops",
    version="1.0",
    ext_modules=[
        NpuExtension(
            name="concat_direct_test_ops",
            sources=["./extension/direct_concat.cpp"],
            extra_compile_args=[
                "-I" + os.path.join(torch_npu_root, "include/third_party/acl/inc"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
