#!/bin/bash
#SBATCH -J resnet_cifar10          # Job name
#SBATCH -o slurm-%j.out            # Name of stdout output file (%j expands to jobId)
#SBATCH --account=GOV108032        # iService Project id
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of MPI process per node
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2               # Number of GPUs per node
#SBATCH --partition=gp1d           # gtest, gp1d, gp2d, p4d

#srun hostname
echo "hostname is $HOSTNAME"

module purge
module load compiler/gnu/7.3.0 openmpi3

export NCCL_DEBUG=INFO
export TF_ENABLE_AUTO_MIXED_PRECISION=1
export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1

SIF=/work/TWCC_cntr/tensorflow_20.09-tf2-py3.sif
SINGULARITY="singularity run --nv $SIF"

#NSYS="nsys profile -t cuda,nvtx,cudnn,mpi --mpi-impl=openmpi -w true -o 2gpu1node"
NSYS=""
cmd="python resnet_cifar.py --num_gpus 2 --batch_size 128 --num_epochs 10"

srun --mpi=pmix $SINGULARITY $NSYS $cmd
