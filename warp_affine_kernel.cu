#include "warp_affine_kernel.h"
#include <cuda_runtime.h>
#include <algorithm>

void AffineMatrix::invertAffineTransform(float imat[6], float omat[6]) {
    float i00 = imat[0], i01 = imat[1], i02 = imat[2];
    float i10 = imat[3], i11 = imat[4], i12 = imat[5];

    float D = i00 * i11 - i01 * i10;
    D = D != 0 ? 1.0f / D : 0;

    float A11 = i11 * D, A12 = -i01 * D;
    float A21 = -i10 * D, A22 = i00 * D;
    float b1 = -A11 * i02 - A12 * i12;
    float b2 = -A21 * i02 - A22 * i12;

    omat[0] = A11; omat[1] = A12; omat[2] = b1;
    omat[3] = A21; omat[4] = A22; omat[5] = b2;
}

void AffineMatrix::compute(const MySize& from, const MySize& to) {
    float scale_x = to.width / (float)from.width;
    float scale_y = to.height / (float)from.height;
    float scale = std::min(scale_x, scale_y);

    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * from.width * 0.5f + to.width * 0.5f + scale * 0.5f - 0.5f;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * from.height * 0.5f + to.height * 0.5f + scale * 0.5f - 0.5f;

    invertAffineTransform(i2d, d2i);
}

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y) {
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value, AffineMatrix matrix
) {
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;

    if (dx >= dst_width || dy >= dst_height) return;

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x=0, src_y=0;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);
    //direct use the affine matrix, structure var
    // src_x = matrix.d2i[0] * dx + matrix.d2i[1] * dy + matrix.d2i[2];
    // src_y = matrix.d2i[3] * dx + matrix.d2i[4] * dy + matrix.d2i[5];

    if (src_x >= -1 && src_x < src_width && src_y >= -1 && src_y < src_height) {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {fill_value, fill_value, fill_value};

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly, hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        // uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
        // uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
        // uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
        // uint8_t* v4 = src + y_high * src_line_size + x_high * 3;

        uint8_t* v1 = const_values;//src + y_low * src_line_size + x_low * 3;
        uint8_t* v2 = const_values;//src + y_low * src_line_size + x_high * 3;
        uint8_t* v3 = const_values;//src + y_high * src_line_size + x_low * 3;
        uint8_t* v4 = const_values;//src + y_high * src_line_size + x_high * 3;
        
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        // c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        // c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        // c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
        // 该点的像素值
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    // uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
    // pdst[0] = static_cast<uint8_t>(c0);
    // pdst[1] = static_cast<uint8_t>(c1);
    // pdst[2] = static_cast<uint8_t>(c2);
    uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
    pdst[0] = c0; pdst[1] = c1; pdst[2] = c2;
}

void warp_affine_bilinear(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value, AffineMatrix matrix
) {
    dim3 block(32, 32);
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    // AffineMatrix affine;
    // affine.compute(MySize(src_width, src_height), MySize(dst_width, dst_height));

    warp_affine_bilinear_kernel<<<grid, block>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, matrix
    );
}

// OpenCV接口函数
cv::Mat warpaffine_gpu(const cv::Mat& ori_image, int dst_height, int dst_width, const AffineMatrix& d2i) {
    cv::Mat output_image(dst_height, dst_width, CV_8UC3);
    uint8_t *psrc_device = nullptr, *pdst_device = nullptr;
    size_t src_size = ori_image.cols * ori_image.rows * 3;
    size_t dst_size = dst_width * dst_height * 3;

    cudaMalloc(&psrc_device, src_size);
    cudaMalloc(&pdst_device, dst_size);
    cudaMemcpy(psrc_device, ori_image.data, src_size, cudaMemcpyHostToDevice);

    //fill value 114
    warp_affine_bilinear(
        psrc_device, ori_image.cols * 3, ori_image.cols, ori_image.rows,
        pdst_device, dst_width * 3, dst_width, dst_height, 114, d2i
    );

    cudaMemcpy(output_image.data, pdst_device, dst_size, cudaMemcpyDeviceToHost);
    cudaFree(psrc_device);
    cudaFree(pdst_device);

    return output_image;
}