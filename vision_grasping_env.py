#!/usr/bin/env python3
"""
Vision-based Grasping Environment for H1 Robot with Allegro Hand
Uses RGB camera input + proprioception + contact sensors
"""

import gymnasium as gym
import numpy as np
import mujoco
# import cv2  # Not needed for now
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

class VisionGraspingEnv(gym.Env):
    """
    Vision-based grasping environment for H1 robot with Allegro hand
    """
    
    def __init__(self, 
                 model_path: str = "scene_h1_allegro_new.xml",
                 camera_name: str = "head_camera",
                 camera_width: int = 640,
                 camera_height: int = 480,
                 max_episode_steps: int = 500,
                 render_mode: Optional[str] = None):
        """
        Initialize the vision-based grasping environment
        
        Args:
            model_path: Path to MuJoCo XML file
            image_size: (width, height) for camera images
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.model_path = model_path
        self.camera_name = camera_name
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Camera setup
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if self.camera_id == -1:
            raise ValueError(f"Camera '{self.camera_name}' not found in model")
        
        # Renderer for camera
        self.renderer = mujoco.Renderer(self.model, self.camera_height, self.camera_width)
        
        # Get joint information for right hand
        self.right_hand_joints = self._get_right_hand_joints()
        self.n_joints = len(self.right_hand_joints)
        
        # Get contact sensor information
        self.contact_sensors = self._get_contact_sensors()
        self.n_contact_sensors = len(self.contact_sensors)
        
        # Get object information
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        if self.object_body_id == -1:
            raise ValueError("Object body not found in model")
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Progressive reward tracking
        self.orientation_achieved = False
        self.approach_achieved = False
        self.opposing_grasp_achieved = False
        self.full_closure_achieved = False
        self.lift_achieved = False
        
        # Penalty tracking
        self.object_knocked_over = False
        self.collision_count = 0
        self.last_object_pos = None
        self.last_palm_pos = None
        
        # Store initial object position for relative measurements
        self.initial_object_height = None
        
        print(f" Vision Grasping Environment initialized")
        print(f"    Camera: {self.camera_name} ({self.camera_width}x{self.camera_height})")
        print(f"    Right hand joints: {self.n_joints}")
        print(f"    Contact sensors: {self.n_contact_sensors}")
        print(f"    Object body ID: {self.object_body_id}")
    
    def _get_right_hand_joints(self) -> list:
        """Get right arm + hand actuator names and IDs (for control)"""
        actuator_names = [
            # Arm joints (for reaching)
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow', 'right_wrist_yaw',
            # Hand joints (for grasping)
            'rh_ffa0', 'rh_ffa1', 'rh_ffa2', 'rh_ffa3',  # Index finger
            'rh_mfa0', 'rh_mfa1', 'rh_mfa2', 'rh_mfa3',  # Middle finger
            'rh_rfa0', 'rh_rfa1', 'rh_rfa2', 'rh_rfa3',  # Ring finger
            'rh_tha0', 'rh_tha1', 'rh_tha2', 'rh_tha3'   # Thumb
        ]
        
        actuator_ids = []
        for actuator_name in actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id != -1:
                actuator_ids.append(actuator_id)
            else:
                print(f"  Warning: Actuator '{actuator_name}' not found")
        
        return actuator_ids
    
    def _get_contact_sensors(self) -> list:
        """Get contact sensor names and IDs"""
        sensor_names = [
            'palm_contact', 'index_contact', 'middle_contact', 
            'ring_contact', 'thumb_contact'
        ]
        
        sensor_ids = []
        for sensor_name in sensor_names:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor_id != -1:
                sensor_ids.append(sensor_id)
            else:
                print(f"  Warning: Sensor '{sensor_name}' not found")
        
        return sensor_ids
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        
        # Action space: Right arm + hand joint positions (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        
        # Observation space components:
        # 1. RGB image: (height, width, channels)
        image_space = spaces.Box(
            low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8
        )
        
        # 2. Proprioception: joint positions + velocities
        proprio_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_joints * 2,), dtype=np.float32
        )
        
        # 3. Contact sensors: binary contact detection
        contact_space = spaces.Box(
            low=0, high=1, shape=(self.n_contact_sensors,), dtype=np.float32
        )
        
        # 4. Object state: position (x, y, z)
        object_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        
        # Combined observation space
        self.observation_space = spaces.Dict({
            'image': image_space,
            'proprioception': proprio_space,
            'contact': contact_space,
            'object_pos': object_space
        })
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        
        # 1. Camera image
        self.renderer.update_scene(self.data, camera=self.camera_id)
        image = self.renderer.render()
        
        # 2. Proprioception (joint positions + velocities)
        # Get actual joint IDs for proprioception (not actuator IDs)
        joint_names = [
            # Arm joints
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow', 'right_wrist_yaw',
            # Hand joints
            'ffj0_right', 'ffj1_right', 'ffj2_right', 'ffj3_right',  # Index finger
            'mfj0_right', 'mfj1_right', 'mfj2_right', 'mfj3_right',  # Middle finger
            'rfj0_right', 'rfj1_right', 'rfj2_right', 'rfj3_right',  # Ring finger
            'thj0_right', 'thj1_right', 'thj2_right', 'thj3_right'   # Thumb
        ]
        
        joint_ids = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                joint_ids.append(joint_id)
        
        joint_positions = self.data.qpos[joint_ids]
        joint_velocities = self.data.qvel[joint_ids]
        proprioception = np.concatenate([joint_positions, joint_velocities])
        
        # 3. Contact sensors (binary: 1 if contact, 0 if no contact)
        contact_values = self.data.sensordata[self.contact_sensors]
        contact_binary = (contact_values > 0.5).astype(np.float32)
        
        # 4. Object position
        object_pos = self.data.xpos[self.object_body_id].copy()
        
        return {
            'image': image,
            'proprioception': proprioception.astype(np.float32),
            'contact': contact_binary,
            'object_pos': object_pos.astype(np.float32)
        }
    
    def _get_reward(self) -> float:
        """Calculate comprehensive progressive reward with penalties"""
        
        # Get current positions
        object_pos = self.data.xpos[self.object_body_id]
        object_height = object_pos[2]
        
        # Get palm position
        palm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm_right")
        if palm_body_id == -1:
            palm_pos = np.array([0.0, 0.0, 0.0])  # Fallback
        else:
            palm_pos = self.data.xpos[palm_body_id]
        
        # Get contact sensor values
        contact_values = self.data.sensordata[self.contact_sensors]
        contact_binary = (contact_values > 0.5).astype(bool)
        
        # Calculate distances and angles
        palm_to_object_distance = np.linalg.norm(palm_pos - object_pos)
        
        # Initialize reward
        reward = 0.0
        
        # === BASIC REWARDS (Always Active) ===
        
        # Distance reward: encourage getting closer to object
        distance_reward = -palm_to_object_distance * 2.0  # Stronger negative reward for distance
        reward += distance_reward
        
        # === PROGRESSIVE REWARDS ===
        
        # 1. Hand Orientation Reward (parallel to object)
        if not self.orientation_achieved:
            # Simple orientation check: palm facing object
            palm_to_object_vector = object_pos - palm_pos
            palm_to_object_distance_normalized = palm_to_object_vector / (np.linalg.norm(palm_to_object_vector) + 1e-6)
            
            # Check if palm is roughly parallel to object (simplified)
            orientation_reward = max(0, 1.0 - palm_to_object_distance / 0.3)  # Within 30cm
            reward += orientation_reward * 0.5  # Reduced continuous orientation reward
            
            if orientation_reward > 0.7:  # Stricter threshold
                reward += 5.0  # Orientation bonus
                self.orientation_achieved = True
        
        # 2. Approach Reward (getting closer)
        if self.orientation_achieved and not self.approach_achieved:
            approach_reward = max(0, 1.0 - palm_to_object_distance / 0.2)  # Within 20cm
            reward += approach_reward * 2.0
            
            if palm_to_object_distance < 0.1:  # Within 10cm
                reward += 10.0  # Approach bonus
                self.approach_achieved = True
        
        # 3. Opposing Grasp Setup (HIGH REWARD)
        if self.approach_achieved and not self.opposing_grasp_achieved:
            # Check for thumb + one finger contact (opposing sides)
            thumb_contact = contact_binary[4]  # thumb_contact
            finger_contacts = contact_binary[1:4]  # index, middle, ring
            
            if thumb_contact and np.any(finger_contacts):
                reward += 25.0  # Opposing grasp bonus
                self.opposing_grasp_achieved = True
        
        # 4. Full Hand Closure
        if self.opposing_grasp_achieved and not self.full_closure_achieved:
            num_contacts = np.sum(contact_binary)
            if num_contacts >= 3:  # Palm + 2+ fingers
                reward += 15.0  # Full closure bonus
                self.full_closure_achieved = True
        
        # 5. Lift Success
        if self.full_closure_achieved and not self.lift_achieved:
            if self.initial_object_height is not None:
                if object_height > self.initial_object_height + 0.05:  # 5cm above starting position
                    reward += 50.0  # Lift bonus
                    self.lift_achieved = True
        
        # === PENALTIES ===
        
        # 1. Object Knockover Penalty (LARGE)
        if self.last_object_pos is not None:
            object_movement = np.linalg.norm(object_pos - self.last_object_pos)
            if object_movement > 0.05 and not self.opposing_grasp_achieved:  # Object moved significantly
                reward -= 20.0  # Knockover penalty
                self.object_knocked_over = True
        
        # 2. Collision Penalties
        # Check for excessive contact (collision)
        if np.sum(contact_binary) > 0 and not self.approach_achieved:
            reward -= 5.0  # Early contact penalty
        
        # 3. Inefficient Movement Penalties
        if self.last_palm_pos is not None:
            palm_velocity = np.linalg.norm(palm_pos - self.last_palm_pos)
            if palm_velocity > 0.1:  # Too fast movement
                reward -= 1.0  # Velocity penalty
            
            # Penalty for moving away from object (common grasping issue)
            last_distance = np.linalg.norm(self.last_palm_pos - object_pos)
            current_distance = palm_to_object_distance
            if current_distance > last_distance + 0.01:  # Moved 1cm away
                reward -= 1.0  # Reduced moving away penalty
        
        # 4. Contact Quality Penalties
        if np.sum(contact_binary) == 1 and contact_binary[0]:  # Only palm contact
            reward -= 3.0  # Palm-only penalty
        
        # 5. Unstable Grasp Penalties
        if self.lift_achieved and self.initial_object_height is not None:
            if object_height < self.initial_object_height - 0.05:  # Object dropped below starting position
                reward -= 30.0  # Drop penalty
        
        # Update tracking variables
        self.last_object_pos = object_pos.copy()
        self.last_palm_pos = palm_pos.copy()
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        
        # Get current object position
        object_pos = self.data.xpos[self.object_body_id]
        object_height = object_pos[2]
        
        # Check if object fell significantly below starting position
        if self.initial_object_height is not None:
            if object_height < self.initial_object_height - 0.2:  # Fell 20cm below start
                return True
        
        # Check if object was successfully lifted (relative to starting position)
        if self.initial_object_height is not None:
            if object_height > self.initial_object_height + 0.1:  # Lifted 10cm above start
                return True
        
        # Check if object was knocked over (large movement)
        if self.object_knocked_over:
            return True
        
        # Check episode length
        if self.episode_step >= self.max_episode_steps:
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Reset progressive reward tracking
        self.orientation_achieved = False
        self.approach_achieved = False
        self.opposing_grasp_achieved = False
        self.full_closure_achieved = False
        self.lift_achieved = False
        
        # Reset penalty tracking
        self.object_knocked_over = False
        self.collision_count = 0
        self.last_object_pos = None
        self.last_palm_pos = None
        
        # Step simulation a few times to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Record initial object height for relative measurements (after stabilization)
        initial_object_pos = self.data.xpos[self.object_body_id]
        self.initial_object_height = initial_object_pos[2]
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step"""
        
        # Convert action to joint positions with coordinated arm control
        joint_actions = action.copy()
        
        # Coordinated arm movements to work with kinematics
        # Action[0] controls arm forward/backward (shoulder pitch only)
        # Action[1] controls arm height (shoulder roll)
        # Action[2] controls arm side movement (shoulder yaw)
        # Action[3] controls wrist orientation
        # Action[4] controls wrist rotation
        # Action[5] controls elbow (only for lifting after grasp)
        
        # Arm forward/backward movement (shoulder pitch only)
        joint_actions[0] = action[0] * 0.4  # shoulder_pitch (increased)
        
        # Arm height control
        joint_actions[1] = action[1] * 0.4  # shoulder_roll (increased)
        
        # Arm side movement (this worked best in tests)
        joint_actions[2] = action[2] * 0.4  # shoulder_yaw (increased)
        
        # Wrist control
        joint_actions[4] = action[4] * 0.5  # wrist_yaw (increased)
        
        # Elbow control (only for lifting after successful grasp)
        # Check if we have contact with object before activating elbow
        contact_values = self.data.sensordata[self.contact_sensors]
        has_contact = np.any(contact_values > 0.5)
        
        if has_contact:
            # Only use elbow for lifting after grasp
            joint_actions[3] = action[5] * 0.3  # elbow (mapped from action[5])
        else:
            # Keep elbow neutral during reaching phase
            joint_actions[3] = 0.0
        
        # Normal scaling for hand joints (6-20, since action[5] is now elbow)
        joint_actions[6:21] = action[6:21] * 1.0  # Hand joints: normal movements
        
        # Apply actions to right hand actuators
        for i, actuator_id in enumerate(self.right_hand_joints):
            # actuator_id is the ctrl index for this actuator
            if actuator_id < len(self.data.ctrl):
                # Ensure action is a scalar value
                action_value = float(joint_actions[i])
                self.data.ctrl[actuator_id] = action_value
            else:
                print(f"  Warning: Actuator ID {actuator_id} out of bounds for ctrl array (size: {len(self.data.ctrl)})")
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        truncated = False
        
        # Update episode tracking
        self.episode_step += 1
        self.episode_reward += reward
        
        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'orientation_achieved': self.orientation_achieved,
            'approach_achieved': self.approach_achieved,
            'opposing_grasp_achieved': self.opposing_grasp_achieved,
            'full_closure_achieved': self.full_closure_achieved,
            'lift_achieved': self.lift_achieved,
            'success': self.lift_achieved,
            'object_knocked_over': self.object_knocked_over
        }
        
        return observation, reward, done, truncated, info
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """Render environment"""
        if mode == 'rgb_array':
            return self._get_observation()['image']
        return None
    
    def close(self):
        """Close environment"""
        if hasattr(self, 'renderer'):
            self.renderer.close()

# Test the environment
if __name__ == "__main__":
    print("🧪 Testing Vision Grasping Environment...")
    
    env = VisionGraspingEnv()
    
    # Test reset
    obs, info = env.reset()
    print(f" Environment reset successful")
    print(f"    Image shape: {obs['image'].shape}")
    print(f"    Proprioception shape: {obs['proprioception'].shape}")
    print(f"    Contact shape: {obs['contact'].shape}")
    print(f"    Object position: {obs['object_pos']}")
    
    # Test step
    action = np.random.uniform(-1, 1, size=env.action_space.shape)
    obs, reward, done, truncated, info = env.step(action)
    print(f" Environment step successful")
    print(f"    Reward: {reward:.3f}")
    print(f"    Info: {info}")
    
    env.close()
    print(" Environment test completed!")
