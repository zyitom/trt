#ifndef WARP_AFFINE_KERNEL_CUH
#define WARP_AFFINE_KERNEL_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    uint8_t* src,
    int src_line_size,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height,
    uint8_t const_value_st,
    float* warp_affine_matrix_2_3,
    bool norm,
    int edge
);

void launchWarpAffineKernel(
    uint8_t* src,
    int src_line_size,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height,
    uint8_t const_value_st,
    float* warp_affine_matrix_2_3,
    bool norm,
    int edge,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
);
#endif // WARP_AFFINE_KERNEL_CUH