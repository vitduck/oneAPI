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

# double precision emulation 
export OverrideDefaultFP64Settings=1
export IGC_EnableDPEmulation=1

src=../sycl

# prepare src
cd $PBS_O_WORKDIR
cp $src/{main.cpp,Makefile} . 

# loop over WG_SIZE
for size in 16 32 64 128
do 
   make WG_SIZE=$size
   echo

   # loop over devices 
   # for device in cpu gpu acc
   for device in gpu
   do
      export SYCL_DEVICE_FILTER=opencl:$device 
     
      for i in 1 2 3
      do 
         (time -p ./lid-wg_${size}.x) 2>&1
	 echo
      done 
      echo 
   done

   make clean
done
