#!/bin/bash -l
#PBS -N JobName
#PBS -l walltime=0:30:00
#PBS -q qcr-users
#PBS -l mem=64gb
#PBS -l ngpus=1
#PBS -j oe

cd $PBS_O_WORKDIR
pixi install
