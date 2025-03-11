#!/bin/bash
#SBATCH --job-name=H2o
#SBATCH --partition=kill-shared
## 3 day max run time for community.q, kill.q, exclusive.q, and htc.q. 1 Hour max run time for sb.q
#SBATCH --time=1-00:00:00 ## time format is DD-HH:MM:SS

#SBATCH --cpus-per-task=12
#SBATCH --mem=80G ## max amount of memory per node you require

#SBATCH --gres=gpu:NV-RTX5000:1  ## request any available gpu
### To request only 1 of the two GPUs in the node, you would do: gpu:NV-K40:1
## GPU TYPES:
#> sinfo -o %20G
## GRES
# gpu:NV-V100-SXM2:8
# gpu:NV-RTX5000:8
# gpu:NV-L40:2
# gpu:NV-H100:1
# gpu:NV-A30:2
# gpu:NV-RTX-A4000:10
# gpu:NV-RTX2080Ti:2
# gpu:NV-RTX2080Ti:8
# gpu:NV-RTX2080Ti:7
# gpu:NV-RTX2070:8

#SBATCH --error=gpu-test-%A.err ## %A - filled with jobid
#SBATCH --output=gpu-test-%A.out ## %A - filled with jobid

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=vchan26@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
module purge
module load lang/Anaconda3
source activate h2o

python h2omodel.py
