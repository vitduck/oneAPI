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

echo "Native CUDA:"
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
( time -p ./snake.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo

echo 

echo "SYCL CUDA:"
( time -p ./snake-K40.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
( time -p ./snake-K40.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
( time -p ./snake-K40.x 100 ../data/ERR240727_1_E2_10million.txt 10000000 ) 2>&1; echo
