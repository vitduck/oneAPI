#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cublas_v2.h>

#include "util.hpp"

#define SEED 666

int SIZE = 4096; 
int LOOP = 100; 

int main(int argc, char *argv[]) {
    // getopt
    parseArguments(argc, argv); 

    // scalar multiplier
    float alpha = 1.0, beta = 1.0;

    // matrix size (squared) 
    int m = SIZE, n = SIZE, k = SIZE; 

    // leading dimension 
    int ldA = k , ldB = n, ldC = n;  

    // host data
    float* A = (float *) aligned_alloc(32, (m * k) * sizeof(float));
    float* B = (float *) aligned_alloc(32, (k * n) * sizeof(float));
    float* C = (float *) aligned_alloc(32, (m * n) * sizeof(float));

    // create random square matrix 
    srand(SEED); 
    random_matrix<float>(A, m, k); 
    random_matrix<float>(B, k, n); 
    zero_matrix<float>(C, m, n); 

    // device data
    float *dA, *dB, *dC; 
    cudaMalloc((void**) &dA, (m * k) * sizeof(float));
    cudaMalloc((void**) &dB, (k * n) * sizeof(float));
    cudaMalloc((void**) &dC, (m * n) * sizeof(float));

    // copy matrix to gpu
    cublasSetMatrix(m, k, sizeof(float), A, ldA, dA, ldA); 
    cublasSetMatrix(k, n, sizeof(float), B, ldB, dB, ldB); 
    cublasSetMatrix(m, n, sizeof(float), C, ldC, dC, ldC); 
    
    // cublas context
    cublasStatus_t status; 
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuda events 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup 
    status = cublasSgemm(
        handle, 
        CUBLAS_OP_T, CUBLAS_OP_T,
        m, n, k, 
        &alpha, dA, ldA, 
        dB, ldB, 
        &beta, dC, ldC
    ); 
    
    // real measurement 
    std::cout << "SGEMM using native cuBLAS" << std::endl; 
    std::cout << "Matrix size: " << SIZE << std::endl; 
    std::cout << "Loop count: "  << LOOP << std::endl; 

    cudaEventRecord(start); 
    for (int i=0; i < LOOP; i++) { 
        status = cublasSgemm(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k, 
            &alpha, dA, ldA, 
            dB, ldB, 
            &beta, dC, ldC
        ); 
    } 
    cudaEventRecord(stop);
        
    // copy matrix back to host
    cublasGetMatrix(m, n, sizeof(float), dC, ldC, C, ldC); 

    cublasDestroy(handle); 
    
    // walltime 
    float walltime = 0; 
    cudaEventElapsedTime(&walltime, start, stop);

    // gflops 
    float average = 0.001*walltime/LOOP;  
    float gflops  = 2.0*m*n*k*1E-9/average; 

    std::cout << "Average running time: " << average << std::endl; 
    std::cout << "Performance GFLOPS: "   << gflops  << std::endl; 

    free(A); 
    free(B); 
    free(C); 
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    return 0;
}
