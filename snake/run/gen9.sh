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

echo "Device: CPU"
export SYCL_DEVICE_FILTER=opencl:cpu
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo

echo "Device: Gen9"
export SYCL_DEVICE_FILTER=opencl:gpu
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
