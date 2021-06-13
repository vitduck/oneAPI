#!/usr/bin/env bash 

#PBS -N test
#PBS -l nodes=1:gen9:gpu:ppn=2
#PBS -l walltime=24:00:00

# Devices:
# 0: Intel(R) FPGA Emulation Device
# 1: Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz
# 2: Intel(R) UHD Graphics P630 [0x3e96]
# 3: Intel(R) UHD Graphics P630 [0x3e96]
# 4: SYCL host device

cd $PBS_O_WORKDIR

# debug 
export SYCL_PI_TRACE=1

# cpu
device=E2176G
export SYCL_DEVICE_FILTER=opencl:cpu
export DPCPP_CPU_PLACES=threads
export DPCPP_CPU_CU_AFFINITY=close

# threads
for i in 1 2 4 8 12
do 
    export DPCPP_CPU_NUM_CUS=$i

    # work roup size 
    for j in 32 64 128 256
    do 
        echo "=> Threads: $i" 2>&1 
        (time -p ./heartwall-intel-wg_$j.x ../data/test.avi 104) 2>&1 
        echo 2>&1

        mv result.txt result-$device-thread_$i-wg_$j.txt

        sleep 3 
    done 
done

# gpu
device=P630
export SYCL_DEVICE_FILTER=opencl:gpu
for j in 32 64 128 256
do 
    (time -p ./heartwall-intel-wg_$j.x ../data/test.avi 104) 2>&1 
    echo 2>&1

    mv result.txt result-$device-wg_$j.txt

    sleep 3 
done 
