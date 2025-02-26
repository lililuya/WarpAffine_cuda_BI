from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="warp_affine",
    ext_modules=[
        CUDAExtension(
            "warp_affine",
            sources=["warp_affine_bind.cpp", "warp_affine_kernel.cu"],
            include_dirs=[
                "/usr/local/include/opencv4",    # OpenCV 头文件路径
                "/home/aiart/.conda/envs/image/include"
            ],
            libraries=["opencv_core", "opencv_imgproc"],  # 链接的 OpenCV 库
            library_dirs=["/usr/local/lib"],      # OpenCV 库文件路径
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)