# Right Hand Grasping with Vision-Based Reinforcement Learning

A comprehensive reinforcement learning system for robotic grasping using vision, proprioception, and contact sensors. This project implements an enhanced CNN-based policy trained with PPO on HPC clusters with A100 GPUs.

## Overview

This system trains a humanoid robot (H1) with Allegro hands to perform precise grasping tasks using:
- High-resolution RGB camera input (640x480)
- Enhanced CNN with residual blocks and spatial attention
- Proprioceptive feedback from 21 joints (5 arm + 16 hand)
- Contact sensors on palm and fingertips
- Progressive reward shaping for effective learning

## Features

- **Vision Processing**: 640x480 RGB camera with enhanced CNN architecture
- **Multi-Modal Input**: Combines vision, proprioception, and tactile feedback
- **Advanced CNN**: Residual blocks with spatial attention mechanism
- **Contact Sensing**: Binary touch detection on palm and all fingertips
- **Progressive Rewards**: Shaped rewards for orientation, approach, grasp, and lift
- **HPC Optimized**: Designed for A100 GPUs with 80GB memory
- **Professional Code**: Clean, documented, and production-ready

## Architecture

### Enhanced CNN Policy
- **Input**: 640x480 RGB images + 50-dimensional proprioception
- **Vision Features**: 524,288 features (vs 8,192 in lightweight version)
- **Architecture**: Residual blocks + spatial attention + adaptive pooling
- **Output**: 21-dimensional action space (5 arm + 16 hand joints)

### Environment
- **Robot**: H1 humanoid with right Allegro hand
- **Sensors**: Head camera, joint encoders, contact sensors
- **Object**: Cylindrical object for grasping
- **Physics**: MuJoCo simulation with realistic dynamics

## Installation

### Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/Right_hand_grasp.git
cd Right_hand_grasp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_hpc.txt
```

### HPC Deployment (TU Dresden)
```bash
# Connect to HPC
ssh zih-username@login1.alpha.hpc.tu-dresden.de

# Create workspace
ws_allocate -F beegfs -n right-hand-grasp -d 90
cd /beegfs/ws/zih-username/right-hand-grasp

# Upload files
scp -r Right_hand_grasp/ zih-username@login1.alpha.hpc.tu-dresden.de:/beegfs/ws/zih-username/right-hand-grasp/

# Setup environment
cd Right_hand_grasp/
bash setup_hpc.sh

# Submit training job
sbatch submit_tu_dresden.sh
```

## Usage

### Local Testing
```bash
# Test environment
python3 vision_grasping_env.py

# Test policy (requires 8GB+ GPU memory)
python3 enhanced_cnn_policy.py

# Run training (requires A100 GPU)
python3 train_vision_grasping_policy.py
```

### HPC Training
```bash
# Monitor job status
squeue --me

# View training progress
tail -f vision-grasping-*.out

# Check for errors
tail -f vision-grasping-*.err
```

## File Structure

```
Right_hand_grasp/
├── enhanced_cnn_policy.py          # Enhanced CNN with residual blocks
├── vision_grasping_env.py          # RL environment with 640x480 camera
├── train_vision_grasping_policy.py # Main training script
├── requirements_hpc.txt            # Python dependencies
├── setup_hpc.sh                   # HPC setup script
├── submit_tu_dresden.sh           # TU Dresden SLURM job script
├── scene_h1_allegro_new.xml       # Main scene file
├── h1_allegro_fixed.xml           # H1 robot model
├── allegro_right.xml              # Right Allegro hand model
├── allegro_left.xml               # Left Allegro hand model
├── push.xml                       # Object model
├── assets/                        # 3D mesh files
└── README.md                      # This file
```

## Training Parameters

- **Total Timesteps**: 1,000,000
- **Batch Size**: 4096 (optimized for A100)
- **Camera Resolution**: 640x480
- **Vision Features**: 524,288
- **Action Space**: 21 (5 arm + 16 hand joints)
- **Save Interval**: 10,000 steps
- **Log Interval**: 1,000 steps

## Expected Performance

- **Training Time**: 2-4 hours on A100 GPU
- **Memory Usage**: 8-12GB GPU memory
- **Success Rate**: 60-80% after full training
- **Convergence**: Grasping behavior within 50,000 steps

## Hardware Requirements

### Minimum (Local Testing)
- GPU: 8GB+ VRAM (RTX 3080, RTX 4080, etc.)
- RAM: 16GB+
- Storage: 5GB

### Recommended (HPC Training)
- GPU: A100 (80GB VRAM)
- RAM: 64GB+
- Storage: 10GB

## Dependencies

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- MuJoCo 3.0+
- Gymnasium 0.29+
- NumPy, Matplotlib, TQDM

## Citation

If you use this code in your research, please cite:

```bibtex
@software{right_hand_grasp,
  title={Right Hand Grasping with Vision-Based Reinforcement Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Right_hand_grasp}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- MuJoCo physics simulator
- PyTorch deep learning framework
- TU Dresden HPC cluster
- Allegro hand hardware specifications
