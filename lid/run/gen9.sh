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
   # for device in cpu gpu
   for device in gpu
   do
      export SYCL_DEVICE_FILTER=opencl:$device 
      
      for i in 1 2 3
      do 
	(time -p ./lid-wg_${size}.x) 2>&1 
      done

      echo 
   done

   make clean
done
