#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda.h>
#include <cublas_v2.h>

#define SEED 666
#define SIZE 16834
#define LOOP 100

#include "util.hpp"

int main(int argc, char *argv[]) {
    cublasStatus_t status; 

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
    cublasHandle_t handle;
    cublasCreate(&handle);

    // warmup 
    status = cublasSgemm(
        handle, 
        CUBLAS_OP_T, CUBLAS_OP_T,
        m, n, k, 
        &alpha, dA, ldA, 
        dB, ldB, 
        &beta, dC, ldC
    ); 
    // copy matrix to host
    cublasGetMatrix(m, n, sizeof(float), dC, ldC, C, ldC); 

    // real measurement 
    std::cout << "SGEMM using native cuBLAS" << std::endl; 
    std::cout << "Matrix size: " << SIZE << std::endl; 

    auto start = std::chrono::system_clock::now(); 
    for (int i=0; i < LOOP; i++) { 
        // warmup 
        status = cublasSgemm(
            handle, 
            CUBLAS_OP_T, CUBLAS_OP_T,
            m, n, k, 
            &alpha, dA, ldA, 
            dB, ldB, 
            &beta, dC, ldC
        ); 
        // copy matrix to host
        cublasGetMatrix(m, n, sizeof(float), dC, ldC, C, ldC); 
    } 
    auto end = std::chrono::system_clock::now(); 

    cublasDestroy(handle); 

    // walltime 
    std::chrono::duration<float> walltime = end-start;

    // gflops 
    float average = walltime.count()/LOOP;  
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
