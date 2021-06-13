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

# debug
export SYCL_PI_TRACE=1

# cpu
export SYCL_DEVICE_FILTER=opencl:cpu
export DPCPP_CPU_NUM_CUS=24
export DPCPP_CPU_PLACES=threads
export DPCPP_CPU_CU_AFFINITY=close

( time -p ./kmeans-intel.x -o -n 10 -m 10 -l 10 -i ../data/10000_34f.txt   ) 2>&1; echo 2>&1
( time -p ./kmeans-intel.x -o -n 10 -m 10 -l 10 -i ../data/100000_34f.txt  ) 2>&1; echo 2>&1 
( time -p ./kmeans-intel.x -o -n 10 -m 10 -l 10 -i ../data/1000000_34f.txt ) 2>&1; echo 2>&1

# gpu
export SYCL_DEVICE_FILTER=opencl:gpu
( time -p ./kmeans-intel.x -o -n 10 -m 10 -l 10 -i ../data/10000_34f.txt   ) 2>&1; echo 2>&1
( time -p ./kmeans-intel.x -o -n 10 -m 10 -l 10 -i ../data/100000_34f.txt  ) 2>&1; echo 2>&1 
( time -p ./kmeans-intel.x -o -n 10 -m 10 -l 10 -i ../data/1000000_34f.txt ) 2>&1; echo 2>&1
