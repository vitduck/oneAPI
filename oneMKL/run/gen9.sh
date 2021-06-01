#!/usr/bin/env bash 

#PBS -N test
#PBS -l nodes=1:gen9:gpu:ppn=2
#PBS -l walltime=04:00:00

# Devices:
# 0: Intel(R) FPGA Emulation Device
# 1: Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz
# 2: Intel(R) UHD Graphics P630 [0x3e96]
# 3: Intel(R) UHD Graphics P630 [0x3e96]
# 4: SYCL host device

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
