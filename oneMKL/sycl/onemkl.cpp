#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "util.hpp"

#define SEED 666

int SIZE = 4096; 
int LOOP = 100; 

int main(int argc, char *argv[]) {
    // getopt 
    parseArguments(argc, argv); 

    // queue creation 
    sycl::device device(sycl::default_selector{}); 
    sycl::queue  queue(device);

    // device info
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";

    // scalar multiplier
    float alpha = 1.0, beta = 1.0;

    // matrix size (squared) 
    int m = SIZE, n = SIZE, k = SIZE; 

    // leading dimension 
    int ldA = k , ldB = n, ldC = n;  

    // USM 
    float *A_USM = sycl::malloc_shared<float>(m * k, queue);
    float *B_USM = sycl::malloc_shared<float>(k * n, queue);
    float *C_USM = sycl::malloc_shared<float>(m * n, queue);

    // create random square matrix 
    srand(SEED); 
    random_matrix<float>(A_USM, m, k); 
    random_matrix<float>(B_USM, k, n); 
    zero_matrix  <float>(C_USM, m, n); 

    // SYCL events 
    sycl::event              gemm_done; 
    std::vector<sycl::event> gemm_dep;   

    // transpose status of matrices
    // following fortran standard, row major is not supported directly, 
    oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::trans;

    // warm up
    gemm_done = oneapi::mkl::blas::column_major::gemm(
        queue, 
        transA, transB, 
        m, n, k, 
        alpha, A_USM, ldA, 
        B_USM, ldB, 
        beta, C_USM, ldC, 
        gemm_dep
    );
    gemm_done.wait(); 

    // real measurement 
    std::cout << "SGEMM using oneMKL" << std::endl; 
    std::cout << "Matrix size: " << SIZE << std::endl; 
    std::cout << "Loop count: "  << LOOP << std::endl; 
    
    auto start = std::chrono::system_clock::now(); 
    for (int i=0; i < LOOP; i++) { 
        gemm_done = oneapi::mkl::blas::column_major::gemm(
            queue, 
            transA, transB, 
            m, n, k, 
            alpha, A_USM, ldA, 
            B_USM, ldB, 
            beta, C_USM, ldC, 
            gemm_dep
        );
        gemm_done.wait();
    } 
    auto end = std::chrono::system_clock::now(); 

    // walltime 
    std::chrono::duration<float> walltime = end-start;

    // gflops 
    float average = walltime.count()/LOOP;  
    float gflops  = 2.0*m*n*k*1E-9/average; 

    std::cout << "Average running time: " << average << std::endl; 
    std::cout << "Performance GFLOPS: "   << gflops  << std::endl; 
    
    // free memory 
    sycl::free(A_USM, queue);
    sycl::free(B_USM, queue);
    sycl::free(C_USM, queue);

    return 0;
}
