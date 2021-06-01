#!/usr/bin/env bash 

#PBS -N test
#PBS -l nodes=iris_xe_max:quad_gpu:ppn=2
#PBS -l walltime=04:00:00

# Devices:
# 0: Intel(R) FPGA Emulation Device
# 1: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz
# 2: Intel(R) Iris(R) Xe MAX Graphics [0x4905]
# 3: Intel(R) Iris(R) Xe MAX Graphics [0x4905]
# 4: Intel(R) Iris(R) Xe MAX Graphics [0x4905]
# 5: Intel(R) Iris(R) Xe MAX Graphics [0x4905]
# 6: Intel(R) Iris(R) Xe MAX Graphics [0x4905]
# 7: Intel(R) Iris(R) Xe MAX Graphics [0x4905]
# 8: Intel(R) Iris(R) Xe MAX Graphics [0x4905]
# 9: Intel(R) Iris(R) Xe MAX Graphics [0x4905]
#10: SYCL host device

cd $PBS_O_WORKDIR

export LD_LIBRARY_PATH=/home/u66264/apps/build/oneMKL/lib:$LD_LIBRARY_PATH

export SYCL_DEVICE_FILTER=opencl:cpu
./sgemm_onemkl.x -s  1024; echo
./sgemm_onemkl.x -s  2048; echo
./sgemm_onemkl.x -s  4096; echo
./sgemm_onemkl.x -s  8192; echo
./sgemm_onemkl.x -s 16384; echo

export SYCL_DEVICE_FILTER=opencl:gpu
./sgemm_onemkl.x -s  1024; echo
./sgemm_onemkl.x -s  2048; echo
./sgemm_onemkl.x -s  4096; echo
./sgemm_onemkl.x -s  8192; echo 
./sgemm_onemkl.x -s 16384; echo
