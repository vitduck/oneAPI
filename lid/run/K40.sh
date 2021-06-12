#!/usr/bin/env bash

#SBATCH -J test
#SBATCH -p ivy_k40_2
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 24:00:00
#SBATCH -e %j.err
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --comment=etc

module purge 
module load gcc/8.3.0 cuda/10.1

# native cuda
for size in 32 64 128 256
do
    (time -p ./lid-cuda-wg_${size}.x) 2>&1
    echo 2>&1 
done

# sycl cuda
export SYCL_PI_TRACE=1
export SYCL_DEVICE_FILTER=cuda:gpu

for size in 32 64 128 256
do
    (time -p ./lid-sycl_K40-wg_${size}.x) 2>&1
    echo 2>&1 
done
