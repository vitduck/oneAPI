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
device=i9
export SYCL_DEVICE_FILTER=opencl:cpu
export DPCPP_CPU_PLACES=threads
export DPCPP_CPU_CU_AFFINITY=close

# threads
for i in 1 2 4 8 12 16 20 24
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
device=iris
export SYCL_DEVICE_FILTER=opencl:gpu

# double prec
export OverrideDefaultFP64Settings=1
export IGC_EnableDPEmulation=1

for j in 32 64 128 256
do 
    (time -p ./heartwall-intel-wg_$j.x ../data/test.avi 104) 2>&1 
    echo 2>&1

    mv result.txt result-$device-wg_$j.txt

    sleep 3 
done 
