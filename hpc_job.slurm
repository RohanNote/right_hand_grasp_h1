#!/bin/bash
#SBATCH --job-name=vision_grasping
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=vision_grasping_%j.out
#SBATCH --error=vision_grasping_%j.err

# Load modules
module load python/3.9
module load cuda/11.8

# Set environment variables
export MUJOCO_GL=egl
export PYTHONPATH=$PYTHONPATH:$PWD

# Create virtual environment
python -m venv hpc_env
source hpc_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mujoco gymnasium numpy matplotlib tqdm

# Run training
echo " Starting Vision-based Grasping Training on A100 GPU"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
echo " Available Memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) MB"

python train_vision_grasping_policy.py

echo " Training completed!"
