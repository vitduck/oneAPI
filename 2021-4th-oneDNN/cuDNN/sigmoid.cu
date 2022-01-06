#include <iostream>
#include <vector> 
#include <algorithm>
#include <cuda_runtime.h> 
#include <cudnn.h>

int main(int argc, char** argv) { 
    // device selection  
    cudaSetDevice(0); 

    // tensor dimension
    int    n = 1, c = 1, h = 1, w = 7;
    int size = n * c * h * w; 

    // host allocation 
    std::vector<float> src(size); 
    std::vector<float> dst(size); 

    // tensor initialization
    for (int i=0; i < size; i++) { src[i] = float(i); }

    // tensor allocation on device
    // can't use std::vector directly in cuDNN
    float *ds, *dd; 
    cudaMalloc(&ds, src.size()*sizeof(float));
    cudaMalloc(&dd, src.size()*sizeof(float));

    // copy src tensor to device memory
    cudaMemcpy(ds, src.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    
    // filter weigth initialization
    const float alpha = 1.0f; 
    const float beta  = 0.0f; 

    // create input tensor descriptor
    cudnnTensorDescriptor_t src_d; 
    cudnnCreateTensorDescriptor(&src_d);
    cudnnSetTensor4dDescriptor(src_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w); 

    // create output tensor descriptor
    cudnnTensorDescriptor_t dst_d;
    cudnnCreateTensorDescriptor(&dst_d);
    cudnnSetTensor4dDescriptor(dst_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w); 

    // create activation descriptor
    cudnnActivationDescriptor_t sigmoid_d;
    cudnnCreateActivationDescriptor(&sigmoid_d);
    cudnnSetActivationDescriptor(sigmoid_d, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0f);

    // create cuda handle
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // sigmoid activation 
    cudnnActivationForward( handle, sigmoid_d, &alpha, src_d, ds, &beta, dst_d, dd); 

    // cleanup 
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(src_d);
    cudnnDestroyTensorDescriptor(dst_d);
    cudnnDestroyActivationDescriptor(sigmoid_d);

    // copy dst tensor to host memory
    cudaMemcpy(dst.data(), dd, size*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "src tensor: "; 
    for (int i=0; i < size; i++) { std::cout << " " << src[i]; } 
    std::cout << std::endl; 
    
    std::cout << "dst tensor: "; 
    for (int i=0; i < size; i++) { std::cout << " " << dst[i]; } 
    std::cout << std::endl; 

    // free memory
    cudaFree(ds); 
    cudaFree(dd); 

    return 0; 
} 
