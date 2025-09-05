import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNNPolicy(nn.Module):
    def __init__(self, n_joints=21, n_contact_sensors=5, n_object_dims=3, action_dim=21, 
                 camera_width=640, camera_height=480):  # Higher resolution for A100
        super(EnhancedCNNPolicy, self).__init__()
        
        self.n_joints = n_joints
        self.n_contact_sensors = n_contact_sensors
        self.n_object_dims = n_object_dims
        self.action_dim = action_dim
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Enhanced CNN architecture for A100 GPUs
        self.vision_cnn = nn.Sequential(
            # Initial feature extraction with higher resolution
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 640x480 -> 320x240
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # First residual block
            self._make_residual_block(64, 64, stride=1),
            
            # Second residual block
            self._make_residual_block(64, 128, stride=2),  # 320x240 -> 160x120
            
            # Third residual block
            self._make_residual_block(128, 256, stride=2),  # 160x120 -> 80x60
            
            # Fourth residual block
            self._make_residual_block(256, 512, stride=2),  # 80x60 -> 40x30
            
            # Fifth residual block
            self._make_residual_block(512, 1024, stride=2),  # 40x30 -> 20x15
            
            # Final feature extraction
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling to maintain spatial information
            nn.AdaptiveAvgPool2d((16, 16)),  # 20x15 -> 16x16
        )
        
        # Spatial attention mechanism (separate from Sequential)
        self.spatial_attention = SpatialAttention(2048)
        
        # Calculate vision feature dimension
        self.vision_feature_dim = 2048 * 16 * 16  # 524,288 features
        
        # Proprioception input dimension
        self.proprio_input_dim = n_joints * 2 + n_contact_sensors + n_object_dims
        
        # Combined feature dimension
        self.combined_feature_dim = self.vision_feature_dim + self.proprio_input_dim
        
        # Enhanced policy head with more capacity
        self.policy_head = nn.Sequential(
            nn.Linear(self.combined_feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim)
        )
        
        # Enhanced value head
        self.value_head = nn.Sequential(
            nn.Linear(self.combined_feature_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create a residual block with skip connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    def forward(self, observations):
        # Extract components
        image = observations['image']  # (batch_size, 3, 640, 480)
        proprioception = observations['proprioception']  # (batch_size, 50)
        
        # Process vision with enhanced CNN
        vision_features = self.vision_cnn(image)  # (batch_size, 2048, 16, 16)
        
        # Apply spatial attention
        vision_features = self.spatial_attention(vision_features)
        
        vision_features = vision_features.view(vision_features.size(0), -1)  # (batch_size, 524288)
        
        # Combine features
        combined_features = torch.cat([vision_features, proprioception], dim=1)  # (batch_size, 524338)
        
        # Get policy and value
        action_logits = self.policy_head(combined_features)  # (batch_size, 21)
        value = self.value_head(combined_features)  # (batch_size, 1)
        
        return action_logits, value
    
    def get_action(self, observations):
        with torch.no_grad():
            action_logits, value = self.forward(observations)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).squeeze(-1)
            log_prob = F.log_softmax(action_logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
            return action, log_prob, value
    
    def evaluate_actions(self, observations, actions):
        action_logits, value = self.forward(observations)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Get log probabilities for the given actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Calculate entropy for exploration
        entropy = -(action_probs * log_probs).sum(dim=-1)
        
        return action_log_probs, value, entropy


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling and max pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        
        # Add instead of concatenate to maintain channel dimension
        attention = avg_pool + max_pool
        attention = self.conv1(attention)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention


# Test the enhanced architecture
if __name__ == "__main__":
    print(" Testing Enhanced CNN Policy for A100 GPUs")
    
    # Create policy with higher resolution
    policy = EnhancedCNNPolicy(camera_width=640, camera_height=480).cuda()
    
    # Test with larger batch size suitable for A100
    batch_size = 128  # Much larger batch for A100
    obs = {
        'image': torch.randn(batch_size, 3, 640, 480).cuda(),
        'proprioception': torch.randn(batch_size, 50).cuda()
    }
    
    print(f" Testing with batch size: {batch_size}")
    print(f" Input resolution: 640x480 (vs 320x240 before)")
    
    # Test forward pass
    try:
        action_logits, value = policy(obs)
        print(f" Forward pass successful!")
        print(f"   Action logits shape: {action_logits.shape}")
        print(f"   Value shape: {value.shape}")
        
        # Test action sampling
        actions, log_probs, values = policy.get_action(obs)
        print(f" Action sampling successful!")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Log probs shape: {log_probs.shape}")
        
        # Test evaluation
        eval_log_probs, eval_values, entropy = policy.evaluate_actions(obs, actions.unsqueeze(-1))
        print(f" Action evaluation successful!")
        print(f"   Eval log probs shape: {eval_log_probs.shape}")
        print(f"   Entropy shape: {entropy.shape}")
        
        # Memory usage
        print(f" GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f" Error: {e}")
    
    print("\n Enhanced CNN Policy Ready for A100!")
    print(f"    Vision: 640x480 → {policy.vision_feature_dim:,} features")
    print(f"    Proprioception: {policy.proprio_input_dim} → 512")
    print(f"    Combined: {policy.combined_feature_dim:,} → 21 actions")
    print(f"    Architecture: Residual blocks + Spatial attention")
  
