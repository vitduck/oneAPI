#include <iostream>
#include <cudnn.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) { 
    // convert src image to matrix format 
    cv::Mat src = cv::imread("./Lenna.png", cv::IMREAD_COLOR);

    // convert src matrix to FP32 with 3 channels
    src.convertTo(src, CV_32FC3); 

    // normalize color range to (0,1)
    cv::normalize(src, src, 0, 1, cv::NORM_MINMAX); 

    // image dimension 
    int channel = src.channels(); 
    int height  = src.rows; 
    int width   = src.cols; 
    int size    = channel * height * width * sizeof(float); 

    // dst vector allocation 
    float *dst = (float*) malloc(size); 
    cv::Mat edge(height, width, CV_32FC3, dst); 

    // filter weigth 
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // edge detection kernel 
    const float laplacian[3][3] = { {1, 1, 1}, {1, -8, 1}, {1, 1, 1} };

    // 4d conv kernel 
    float filter[3][3][3][3];

    // copy laplacian to 4d tensor
    for (int i = 0; i < 3; i++) 
        for (int j = 0; j < 3; j++) 
            for (int k = 0; k < 3; k++) 
                for (int m = 0; m < 3; m++) 
                    filter[i][j][k][m] = laplacian[k][m];

    // device selection  
    cudaSetDevice(0); 

    // create cuda handle
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // create input tensor descriptor
    cudnnTensorDescriptor_t src_d; 
    cudnnCreateTensorDescriptor(&src_d);
    cudnnSetTensor4dDescriptor(src_d, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channel, height, width); 

    // create filter descriptor 
    cudnnFilterDescriptor_t filter_d; 
    cudnnCreateFilterDescriptor(&filter_d); 
    cudnnSetFilter4dDescriptor(filter_d, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, 3, 3, 3); 

    // create convolution descriptor 
    cudnnConvolutionDescriptor_t conv_d; 
    cudnnCreateConvolutionDescriptor(&conv_d); 
    cudnnSetConvolution2dDescriptor(conv_d, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT); 

    // create output tensor descriptor
    cudnnTensorDescriptor_t dst_d; 
    cudnnCreateTensorDescriptor(&dst_d);
    cudnnSetTensor4dDescriptor(dst_d, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channel, height, width); 

    // creat convulution algorithm 
    cudnnConvolutionFwdAlgo_t conv_algo; 
    cudnnGetConvolutionForwardAlgorithm(handle, src_d, filter_d, conv_d, dst_d, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_algo); 

    // get workspace size
    size_t ws_size; 
    cudnnGetConvolutionForwardWorkspaceSize(handle, src_d, filter_d, conv_d, dst_d, conv_algo, &ws_size); 

    // memory allocation for input/output/filter/workspace
    void *ds, *dd, *df, *dw; 
    cudaMalloc(&ds, size);
    cudaMalloc(&dd, size);
    cudaMalloc(&df, size);
    cudaMalloc(&dw, ws_size);

    // copy data to device 
    cudaMemcpy(ds, src.ptr<float>(0), size, cudaMemcpyHostToDevice);
    cudaMemcpy(df, filter, sizeof(filter)*sizeof(float), cudaMemcpyHostToDevice);

    // convolution 
    cudnnConvolutionForward(handle, &alpha, src_d, ds, filter_d, df, conv_d, conv_algo, dw, ws_size, &beta, dst_d, dd); 

    // copy data back to host 
    cudaMemcpy(dst, dd, size, cudaMemcpyDeviceToHost);

    // avoid negative pixels ?! 
    cv::threshold(edge, edge, 0, 0, cv::THRESH_TOZERO);
   
    // renormalize to RBG range 
    cv::normalize(edge, edge, 0.0, 255.0, cv::NORM_MINMAX);

    // convert back to 8bit (RGB) format 
    edge.convertTo(edge, CV_8UC3);

    // write to png file 
    cv::imwrite("detection.png", edge);

    // clean up dnn
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(src_d);
    cudnnDestroyTensorDescriptor(dst_d);
    cudnnDestroyFilterDescriptor(filter_d);
    cudnnDestroyConvolutionDescriptor(conv_d); 

    // free memory
    cudaFree(ds); 
    cudaFree(dd); 
    cudaFree(df); 
    cudaFree(dw); 
}
