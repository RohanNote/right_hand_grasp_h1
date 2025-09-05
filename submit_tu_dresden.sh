#!/bin/bash

#SBATCH -A project            # Choose one group (type groups)
#SBATCH -p alpha              # Specify partition
#SBATCH -J "vision-grasping"  # Set job name
#SBATCH --gres=gpu:1          # Use one GPU (A100)
#SBATCH -N 1                  # Use one node
#SBATCH --cpus-per-task=8     # Use 8 CPU cores
#SBATCH --mem-per-cpu=10G     # Memory per CPU
#SBATCH --mem=80G             # Specify total memory needs
#SBATCH --time=4:00:00        # Set runtime to 4 hours
#SBATCH -o "vision-grasping-%j.out"	# Output log name
#SBATCH -e "vision-grasping-%j.err"	# Error log name

# Print job information
echo "Starting Vision-based Grasping Training on TU Dresden HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
echo "Available Memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) MB"

# Activate the virtual environment
source /beegfs/ws/$USER/vision-grasping/venv/bin/activate

# Load Python module
module load Python/11.5

# Set environment variables
export MUJOCO_GL=egl
export PYTHONPATH=$PYTHONPATH:$PWD

# Navigate to training directory
cd /beegfs/ws/$USER/vision-grasping/HPC_Upload/

# Start training
echo "Starting training at $(date)"
python3 train_vision_grasping_policy.py

# Training completed
echo "Training completed at $(date)"

# Deactivate the virtual environment
deactivate
