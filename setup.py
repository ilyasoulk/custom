from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='custom._C', #
            sources=[
                'csrc/bindings.cpp',
                'csrc/matmul.cu',
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)