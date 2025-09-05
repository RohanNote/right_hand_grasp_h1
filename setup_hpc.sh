#!/bin/bash
# HPC Setup Script for Vision-based Grasping Training
# Optimized for A100 GPUs

echo "Setting up Vision-based Grasping Training on HPC"
echo "Target: A100 GPUs with 80GB memory"

# Check if we're on HPC
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Warning: This script should be run on HPC cluster"
    echo "Please submit as a job or run in interactive session"
fi

# Load required modules
echo "Loading modules..."
module load python/3.9
module load cuda/11.8

# Set environment variables
export MUJOCO_GL=egl
export PYTHONPATH=$PYTHONPATH:$PWD

# Create virtual environment
echo "Creating virtual environment..."
python -m venv hpc_env
source hpc_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements_hpc.txt

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test the enhanced CNN policy
echo "Testing Enhanced CNN Policy..."
python -c "
from enhanced_cnn_policy import EnhancedCNNPolicy
import torch

# Test with A100-appropriate batch size
policy = EnhancedCNNPolicy(camera_width=640, camera_height=480).cuda()
batch_size = 128
obs = {
    'image': torch.randn(batch_size, 3, 640, 480).cuda(),
    'proprioception': torch.randn(batch_size, 50).cuda()
}

action_logits, value = policy(obs)
print(f'Enhanced CNN Policy test successful!')
print(f'   Input resolution: 640x480')
print(f'   Batch size: {batch_size}')
print(f'   Output shapes: actions={action_logits.shape}, value={value.shape}')
print(f'   Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
"

echo "HPC Setup Complete!"
echo "Next steps:"
echo "   1. Submit job: sbatch hpc_job.slurm"
echo "   2. Monitor: squeue -u \$USER"
echo "   3. Check logs: tail -f vision_grasping_*.out"
echo ""
echo " Enhanced features for A100:"
echo "    Higher resolution: 640x480 (vs 320x240)"
echo "    Enhanced CNN: Residual blocks + Spatial attention"
echo "    Larger batch size: 4096 (vs 512)"
echo "    More features: 524K vision features (vs 8K)"
