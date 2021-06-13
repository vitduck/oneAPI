#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include "mkl.h"

#include "util.hpp"

#define SEED 666

int SIZE = 4096; 
int LOOP = 100; 

int main(int argc, char *argv[]) {
    // getopt 
    //parseArguments(argc, argv); 

    // scalar multiplier
    float alpha = 1.0, beta = 1.0;

    // matrix size (squared) 
    int m = SIZE, n = SIZE, k = SIZE; 

    // leading dimension 
    int ldA = k , ldB = n, ldC = n;  

    // MKL aligns array on 64-bytes memory
    float *A = (float*) mkl_malloc(sizeof(float)*m*k, 64); 
    float *B = (float*) mkl_malloc(sizeof(float)*k*n, 64); 
    float *C = (float*) mkl_malloc(sizeof(float)*m*n, 64); 

    // create random square matrix 
    srand(SEED); 
    random_matrix<float>(A, m, k); 
    random_matrix<float>(B, k, n); 
    zero_matrix  <float>(C, m, n); 

    // warm up
    cblas_sgemm(
        CblasColMajor, 
        CblasNoTrans, CblasNoTrans, 
        m, n, k, 
        alpha, A, ldA, B, ldB, beta, C, ldC
    ); 

    auto start = std::chrono::high_resolution_clock::now(); 
    for (int i=0; i < LOOP; i++) { 
        cblas_sgemm(
            CblasColMajor, 
            CblasNoTrans, CblasNoTrans, 
            m, n, k, 
            alpha, A, ldA, B, ldB, beta, C, ldC
         ); 
    } 
    auto end = std::chrono::high_resolution_clock::now();

    // walltime 
    std::chrono::duration<float> walltime = end-start; 

    // gflops 
    float average = walltime.count()/LOOP;  
    float gflops  = 2.0*m*n*k*1E-9/average; 

    std::cout << "Average running time: " << average << std::endl; 
    std::cout << "Performance GFLOPS: "   << gflops  << std::endl; 
    
    // free memory 
    mkl_free(A); 
    mkl_free(B); 
    mkl_free(C); 

    return 0;
}
