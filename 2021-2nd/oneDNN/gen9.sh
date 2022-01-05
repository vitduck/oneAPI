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

export DNNL_VERBOSE=1
export LD_PRELOAD=/home/u66264/apps/build/oneDNN/lib/libdnnl.so

./benchdnn --conv --engine=gpu --mode=P --batch=inputs/conv/perf_conv_gen9
