#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "warp_affine_kernel.h"

namespace py = pybind11;

// 定义输入图像、目标图像尺寸、仿射矩阵的逆矩阵
py::array_t<uint8_t> warpaffine_gpu_py(py::array_t<uint8_t>& input, int dst_height, int dst_width, py::array_t<float>& d2i) {
    
    //检查输入
    py::buffer_info buf = input.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Input image must be a 3-channel (H, W, 3) image.");
    }
    
    // 类型转化
    cv::Mat ori_image(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
    
    // 获取仿射矩阵的 buffer 信息, 防止非法输入
    py::buffer_info d2i_buf = d2i.request();
    if (d2i_buf.size != 6) {
        throw std::runtime_error("d2i must be a 6-element array.");
    }
    float* d2i_ptr = static_cast<float*>(d2i_buf.ptr);

     // 将 d2i_ptr 转换为 AffineMatrix 对象
     AffineMatrix affine;
     for (int i = 0; i < 6; ++i) {
         affine.d2i[i] = d2i_ptr[i];
     }

    cv::Mat output = warpaffine_gpu(ori_image, dst_height, dst_width, affine);
    return py::array_t<uint8_t>({output.rows, output.cols, 3}, output.data);
}

PYBIND11_MODULE(warp_affine, m) {
    m.def("warpaffine_gpu", &warpaffine_gpu_py, "Warp affine on GPU",
    py::arg("input"), py::arg("dst_height"), py::arg("dst_width"), py::arg("d2i"));
}