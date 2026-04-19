#!/bin/bash
#SBATCH --job-name="train_SDFVAE_cod_sharded_dataset_stage1_new_embed_method"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
##BATCH --nodelist=node[11-18,20,21,4,5]
#SBATCH --nodelist=node21
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=END
#SBATCH --output=/home/stud/dyun/makeathon-challenge-2026/logs/slurm/slurm-%j.log
#SBATCH --error=/home/stud/dyun/makeathon-challenge-2026/logs/slurm/slurm-%j.log

# export NCCL_DEBUG=INFO
# export HYDRA_FULL_ERROR=1
module load cuda/12.1.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
source /home/stud/dyun/miniconda3/etc/profile.d/conda.sh

conda activate /home/stud/dyun/miniconda3/envs/cod-fm

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"    
echo "Node(s): $SLURM_NODELIST" 
echo "==== CPU Info ===="
lscpu

echo "==== Memory Info ===="
free -h
echo "GPUs:    $NUM_GPUS  | Threads per proc: $THREADS_PER_PROC"

pwd; hostname; date
nvidia-smi
nvidia-smi topo -m

cd /home/stud/dyun/makeathon-challenge-2026
# python /home/stud/dyun/SDFencoder/helpers/count_surface_points_gpu.py
srun python train.py --config-name config_vit
