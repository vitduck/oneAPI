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
( ./heartwall.x 104 ) 2>&1; echo
( ./heartwall.x 104 ) 2>&1; echo
( ./heartwall.x 104 ) 2>&1; echo

echo "Device: Gen9"
export SYCL_DEVICE_FILTER=opencl:gpu
( ./heartwall.x 104 ) 2>&1; echo
( ./heartwall.x 104 ) 2>&1; echo
( ./heartwall.x 104 ) 2>&1; echo
