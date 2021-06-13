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

echo "Native CUDA:" 2>& 1 
( time -p ./kmeans-cuda.x -o -n 10 -m 10 -l 10 -i ../data/10000_34f.txt ) 2>&1; echo
( time -p ./kmeans-cuda.x -o -n 10 -m 10 -l 10 -i ../data/100000_34f.txt ) 2>&1; echo
( time -p ./kmeans-cuda.x -o -n 10 -m 10 -l 10 -i ../data/1000000_34f.txt ) 2>&1; echo

echo 2>&1

echo "SYCL CUDA:"
export SYCL_PI_TRACE=1
export SYCL_DEVICE_FILTER=cuda:gpu
( time -p ./kmeans-sycl_K40.x -o -n 10 -m 10 -l 10 -i ../data/10000_34f.txt ) 2>&1; echo
( time -p ./kmeans-sycl_K40.x -o -n 10 -m 10 -l 10 -i ../data/100000_34f.txt ) 2>&1; echo
( time -p ./kmeans-sycl_K40.x -o -n 10 -m 10 -l 10 -i ../data/1000000_34f.txt ) 2>&1; echo
