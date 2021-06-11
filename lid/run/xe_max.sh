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

# cpu device
export SYCL_DEVICE_FILTER=opencl:cpu
export DPCPP_CPU_NUM_CUS=24
export DPCPP_CPU_PLACES=threads
export DPCPP_CPU_CU_AFFINITY=close

for size in 32 64 128 256
do
    (time -p ./lid-intel-wg_${size}.x) 2>&1
done

# gpu device
# double precision emulation
export OverrideDefaultFP64Settings=1
export IGC_EnableDPEmulation=1

export SYCL_DEVICE_FILTER=opencl:gpu
for size in 32 64 128 256
do
    (time -p ./lid-intel-wg_${size}.x) 2>&1
done
