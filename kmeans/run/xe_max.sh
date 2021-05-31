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

echo "Device: CPU"
export SYCL_DEVICE_FILTER=opencl:cpu
( time -p ./kmeans.x -r -n 5 -m 15 -l 10 -i ../../data/10000_34f.txt   ) 2>&1; echo
( time -p ./kmeans.x -r -n 5 -m 15 -l 10 -i ../../data/100000_34f.txt  ) 2>&1; echo
( time -p ./kmeans.x -r -n 5 -m 15 -l 10 -i ../../data/1000000_34f.txt ) 2>&1; echo

echo "Device: Xe_Max"
export SYCL_DEVICE_FILTER=opencl:gpu
( time -p ./kmeans.x -r -n 5 -m 15 -l 10 -i ../../data/10000_34f.txt   ) 2>&1; echo
( time -p ./kmeans.x -r -n 5 -m 15 -l 10 -i ../../data/100000_34f.txt  ) 2>&1; echo
( time -p ./kmeans.x -r -n 5 -m 15 -l 10 -i ../../data/1000000_34f.txt ) 2>&1; echo
