#include <iostream>
#include <cuda_runtime.h> 
#include <cudnn.h>

int main(int argc, char** argv) {
    // device selection  
    cudaSetDevice(0); 

    // tensor dimension
    int    n = 1, c = 1, h = 1, w = 7;
    int size = n * c * h * w; 

    // tensor allocation (USM)
    float *src, *dst; 
    cudaMallocManaged(&src, size * sizeof(float));
    cudaMallocManaged(&dst, size * sizeof(float));
    
    // tensor initialization
    for (int i=0; i < size; i++) { src[i] = float(i); }
    for (int i=0; i < size; i++) { dst[i] = 0.f; } 

    // filter weigth initialization
    float alpha[1] = {1.0};
    float  beta[1] = {0.0};

    // create input tensor descriptor
    cudnnTensorDescriptor_t src_d, dst_d;
    cudnnCreateTensorDescriptor(&src_d);
    cudnnCreateTensorDescriptor(&dst_d);
    cudnnSetTensor4dDescriptor(src_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w); 
    cudnnSetTensor4dDescriptor(dst_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w); 

    // create activation descriptor
    cudnnActivationDescriptor_t sigmoid_d;
    cudnnCreateActivationDescriptor(&sigmoid_d);
    cudnnSetActivationDescriptor(sigmoid_d, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0f);

    // create cuda handle
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // sigmoid activation 
    cudnnActivationForward(
        handle,
        sigmoid_d,
        alpha,
        src_d,
        src,
        beta,
        dst_d,
        dst
    ); 
    
    // free cuDNN
    data is automatically copied back to host
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(src_d); 
    cudnnDestroyTensorDescriptor(dst_d); 
    cudnnDestroyActivationDescriptor(sigmoid_d); 

    std::cout << "src tensor: "; 
    for (int i=0; i < size; i++) { std::cout << " " << src[i]; } 
    std::cout << std::endl; 
    
    std::cout << "dst tensor: "; 
    for (int i=0; i < size; i++) { std::cout << " " << dst[i]; } 
    std::cout << std::endl; 

    // free memory
    cudaFree(src); 
    cudaFree(dst); 

    return 0; 
}
