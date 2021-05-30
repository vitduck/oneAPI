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

src=../
device=K40

# prepare src
cp $src/{main.cpp,Makefile.cuda} .

# loop over WG_SIZ
for size in 16 32 64 128
do
    make BUILD=$device WG_SIZE=$size -f Makefile.cuda

    echo 

    for i in 1 2 3
    do
        (time -p ./lid-${device}-wg_${size}.x) 2>&1
        echo 
    done

    echo

    make clean -f Makefile.cuda
done
