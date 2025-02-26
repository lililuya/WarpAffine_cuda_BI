#ifndef WARP_AFFINE_KERNEL_H
#define WARP_AFFINE_KERNEL_H

#include <cstdint>
#include <opencv2/opencv.hpp>

struct MySize {
    int width = 0, height = 0;
    MySize() = default;
    MySize(int w, int h) : width(w), height(h) {}
};

struct AffineMatrix {
    float i2d[6];
    float d2i[6];
    void invertAffineTransform(float imat[6], float omat[6]);
    void compute(const MySize& from, const MySize& to);
};

// 声明CUDA函数
void warp_affine_bilinear(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value
);

// 声明面向Python的接口函数
cv::Mat warpaffine_gpu(const cv::Mat& image, int dst_height, int dst_width, const AffineMatrix& affine);

#endif