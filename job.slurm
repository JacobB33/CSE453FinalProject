#!/bin/bash

#SBATCH --job-name=jacob-experiment
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jacob33@cs.washington.edu

#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH --nodes=12
#SBATCH --ntasks=12

#SBATCH --gpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/jacob33/CSE453FinalProject
#SBATCH --export=all
#SBATCH --output=slurm/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm/%j-err.txt    # where STDERR goes

# Activate Anaconda work environment for OpenDrift
source /mmfs1/home/jacob33/miniconda3/bin/activate
conda activate /gscratch/weirdlab/jacob33/envs


# Your programs to run.
export WANDB_MODE=online
export WANDB_PROJECT=CSE-453-project

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 12 \
--nproc_per_node 4  \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29510 \
run_training.py
# remove the checkpoint since we do not care
# rm *.pt
