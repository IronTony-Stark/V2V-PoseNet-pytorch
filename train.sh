#!/bin/bash

#current_datetime=$(date +"%Y%m%d_%H%M%S")

#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:tesla
###SBATCH --gres=gpu:a100-40g
#SBATCH --time=4000
###SBATCH --output=output_${current_datetime}.log
###SBATCH --error=error_${current_datetime}.log
#SBATCH --job-name=ironanton
### --nodelist=falcon5

module load any/python/3.8.3-conda

CONDA_ENV_PATH=/gpfs/space/home/zaliznyi/miniconda3/envs/v2vposenet/

conda info --envs
conda activate $CONDA_ENV_PATH

nvidia-smi
gcc --version

export PYTHONPATH=/gpfs/space/home/zaliznyi/projects/V2V-PoseNet-pytorch

#$CONDA_ENV_PATH/bin/python ./experiments/msra-subject3/main.py
#$CONDA_ENV_PATH/bin/python ./experiments/msra-subject3/gen_gt.py
#$CONDA_ENV_PATH/bin/python ./experiments/msra-subject3/show_acc.py

$CONDA_ENV_PATH/bin/python ./experiments/toy-data-keypoints/main.py
