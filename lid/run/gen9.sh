#!/usr/bin/env bash 

#PBS -N test
#PBS -l nodes=1:gen9:gpu:ppn=2
#PBS -l walltime=01:00:00

# Devices:
# 0: Intel(R) FPGA Emulation Device
# 1: Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz
# 2: Intel(R) UHD Graphics P630 [0x3e96]
# 3: Intel(R) UHD Graphics P630 [0x3e96]
# 4: SYCL host device

cd $PBS_O_WORKDIR

# debug 
export SYCL_PI_TRACE=1

# cpu device 
export SYCL_DEVICE_FILTER=opencl:cpu
export DPCPP_CPU_NUM_CUS=12
export DPCPP_CPU_PLACES=threads
export DPCPP_CPU_CU_AFFINITY=close

for size in 32 64 128 256
do 
    (time -p ./lid-intel-wg_${size}.x) 2>&1 
done 

# gpu device 
export SYCL_DEVICE_FILTER=opencl:gpu
for size in 32 64 128 256
do 
    (time -p ./lid-intel-wg_${size}.x) 2>&1 
done 
