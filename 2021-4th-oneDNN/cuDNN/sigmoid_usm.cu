#include <iostream>
#include <cuda_runtime.h> 
#include <cudnn.h>

int main(int argc, char** argv) {
    // device selection  
    cudaSetDevice(0); 

    // tensor dimension
    const int N = 1, C = 1, H = 1, W = 7;

    // tensor allocation (USM)
    float *src, *dst; 
    cudaMallocManaged(&src, N*C*H*W*sizeof(float));
    cudaMallocManaged(&dst, N*C*H*W*sizeof(float));
    
    // tensor initialization
    for (int i=0; i < W; i++) { src[i] = float(i); }

    // filter weigth initialization
    const float alpha = 1.0;
    const float  beta = 0.0;

    // create input tensor descriptor
    cudnnTensorDescriptor_t src_d; 
    cudnnCreateTensorDescriptor(&src_d);
    cudnnSetTensor4dDescriptor(src_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W); 

    // create  output tensor descriptor
    cudnnTensorDescriptor_t dst_d; 
    cudnnCreateTensorDescriptor(&dst_d);
    cudnnSetTensor4dDescriptor(dst_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W); 

    // create activation descriptor
    cudnnActivationDescriptor_t sigmoid_d;
    cudnnCreateActivationDescriptor(&sigmoid_d);
    cudnnSetActivationDescriptor(sigmoid_d, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0f);

    // create cuda handle
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // sigmoid activation 
    cudnnActivationForward(handle, sigmoid_d, &alpha, src_d, src, &beta, dst_d, dst); 
    
    // free cuDNN
    // data is automatically copied back to host
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(src_d); 
    cudnnDestroyTensorDescriptor(dst_d); 
    cudnnDestroyActivationDescriptor(sigmoid_d); 

    std::cout << "src tensor: "; 
    for (int i=0; i < W; i++) { std::cout << " " << src[i]; } 
    std::cout << std::endl; 
    
    std::cout << "dst tensor: "; 
    for (int i=0; i < W; i++) { std::cout << " " << dst[i]; } 
    std::cout << std::endl; 

    // free memory
    cudaFree(src); 
    cudaFree(dst); 

    return 0; 
}
