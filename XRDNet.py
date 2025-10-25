"""
XRD-Net: A Novel Deep Learning Architecture for Alzheimer's Disease Classification

This implementation includes:
- Dense blocks for feature reuse
- Residual connections for enhanced gradient flow
- Spatial Context Fusion (SCF) blocks for multi-scale feature extraction
- Integrated attention mechanisms
- Classification head with dropout

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """
    Dense Block with feature concatenation for feature reuse.
    Implements equations (1), (2), and (3) from the paper.
    """
    def __init__(self, in_channels, growth_rate=32, num_layers=3):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.layers.append(layer)
        
        self.num_layers = num_layers
        self.growth_rate = growth_rate
    
    def forward(self, x):
        """
        Forward pass with dense connections.
        Z_l' = concat(Z_l, Z_(l-1))
        """
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        
        return torch.cat(features, dim=1)


class SpatialContextFusionBlock(nn.Module):
    """
    Spatial Context Fusion Block with multi-scale feature extraction
    and attention mechanism.
    Implements equations (4)-(11) from the paper.
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialContextFusionBlock, self).__init__()
        
        # Local branch (3x3 conv for fine-grained features)
        self.local_branch = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=True)
        )
        
        # Global branch (5x5 conv for broader context)
        self.global_branch = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, 
                     stride=1, padding=2, bias=True)
        )
        
        # Attention mechanism projections
        self.query_conv = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(out_channels, out_channels, 
                                  kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=1, bias=False)
        
        # Output projection
        self.output_conv = nn.Conv2d(out_channels * 2, out_channels, 
                                     kernel_size=1, bias=True)
        
        self.out_channels = out_channels
    
    def forward(self, x):
        """
        Forward pass implementing multi-scale fusion with attention.
        """
        # Extract local and global features (Equations 4 & 5)
        F_local = self.local_branch(x)
        F_global = self.global_branch(x)
        
        # Attention mechanism (Equations 6-10)
        batch_size, channels, height, width = F_local.size()
        
        # Project features into Q, K, V
        Q = self.query_conv(F_local)  # Equation 6
        K = self.key_conv(F_global)   # Equation 7
        V = self.value_conv(F_global) # Equation 8
        
        # Reshape for attention computation
        Q = Q.view(batch_size, channels, -1)  # [B, C, H*W]
        K = K.view(batch_size, channels, -1)  # [B, C, H*W]
        V = V.view(batch_size, channels, -1)  # [B, C, H*W]
        
        # Compute attention scores (Equation 9)
        # A_s = softmax(Q * K^T / sqrt(d_c))
        attention_scores = torch.bmm(Q.transpose(1, 2), K)  # [B, H*W, H*W]
        attention_scores = attention_scores / (channels ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values (Equation 10)
        A_out = torch.bmm(V, attention_scores.transpose(1, 2))
        A_out = A_out.view(batch_size, channels, height, width)
        
        # Concatenate and project (Equation 11)
        S_out = torch.cat([A_out, F_local], dim=1)
        S_out = self.output_conv(S_out)
        
        return S_out


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connections for enhanced gradient flow.
    Implements equations (12)-(15) from the paper.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (1x1 conv to match dimensions)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 
                                         kernel_size=1, stride=1, bias=True)
        
        self.relu_out = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        Z_out = ReLU(R_2 + Z_s)
        """
        # Skip connection (Equation 12)
        Z_s = self.skip_connection(x)
        
        # Main path (Equations 13-14)
        R_1 = self.relu1(self.bn1(self.conv1(x)))
        R_2 = self.bn2(self.conv2(R_1))
        
        # Add and activate (Equation 15)
        Z_out = self.relu_out(R_2 + Z_s)
        
        return Z_out


class XRDNet(nn.Module):
    """
    XRD-Net: Complete architecture for Alzheimer's Disease classification.
    
    Architecture components:
    - Initial convolution layer
    - 3 Dense blocks with increasing depth (32, 64, 128)
    - 3 SCF blocks with increasing depth (32, 64, 128)
    - 3 Residual blocks with increasing depth (64, 128, 256)
    - Classification head with GAP, FC layers, and dropout
    
    Args:
        num_classes (int): Number of output classes (e.g., 4 for AD stages)
        growth_rate (int): Growth rate for dense blocks
        dropout_rate (float): Dropout rate for classification head
    """
    def __init__(self, num_classes=4, growth_rate=32, dropout_rate=0.5):
        super(XRDNet, self).__init__()
        
        # Initial convolution block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # First stage: Dense Block 1 (depth=32)
        self.dense_block1 = DenseBlock(in_channels=32, growth_rate=growth_rate, num_layers=3)
        # Output channels: 32 + 3*32 = 128
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.scf_block1 = SpatialContextFusionBlock(in_channels=128, out_channels=32)
        self.residual_block1 = ResidualBlock(in_channels=32, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second stage: Dense Block 2 (depth=64)
        self.dense_block2 = DenseBlock(in_channels=64, growth_rate=growth_rate, num_layers=3)
        # Output channels: 64 + 3*32 = 160
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.scf_block2 = SpatialContextFusionBlock(in_channels=160, out_channels=64)
        self.residual_block2 = ResidualBlock(in_channels=64, out_channels=128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third stage: Dense Block 3 (depth=128)
        self.dense_block3 = DenseBlock(in_channels=128, growth_rate=growth_rate, num_layers=3)
        # Output channels: 128 + 3*32 = 224
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.scf_block3 = SpatialContextFusionBlock(in_channels=224, out_channels=128)
        self.residual_block3 = ResidualBlock(in_channels=128, out_channels=256)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classification head (Equations 16-19)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through XRD-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, 128, 128]
        
        Returns:
            torch.Tensor: Output logits of shape [B, num_classes]
        """
        # Initial convolution
        x = self.initial_conv(x)  # [B, 32, 64, 64]
        
        # Stage 1
        x = self.dense_block1(x)      # [B, 128, 64, 64]
        x = self.pool1(x)             # [B, 128, 32, 32]
        x = self.scf_block1(x)        # [B, 32, 32, 32]
        x = self.residual_block1(x)   # [B, 64, 32, 32]
        x = self.pool2(x)             # [B, 64, 16, 16]
        
        # Stage 2
        x = self.dense_block2(x)      # [B, 160, 16, 16]
        x = self.pool3(x)             # [B, 160, 8, 8]
        x = self.scf_block2(x)        # [B, 64, 8, 8]
        x = self.residual_block2(x)   # [B, 128, 8, 8]
        x = self.pool4(x)             # [B, 128, 4, 4]
        
        # Stage 3
        x = self.dense_block3(x)      # [B, 224, 4, 4]
        x = self.pool5(x)             # [B, 224, 2, 2]
        x = self.scf_block3(x)        # [B, 128, 2, 2]
        x = self.residual_block3(x)   # [B, 256, 2, 2]
        x = self.pool6(x)             # [B, 256, 1, 1]
        
        # Classification head (Equations 16-19)
        x = self.global_avg_pool(x)   # [B, 256, 1, 1]
        x = torch.flatten(x, 1)       # [B, 256]
        x = self.fc1(x)               # [B, 256]
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)               # [B, num_classes]
        
        return x
    
    def extract_features(self, x):
        """
        Extract feature maps before classification for visualization.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, 128, 128]
        
        Returns:
            torch.Tensor: Feature maps of shape [B, 256, 1, 1]
        """
        x = self.initial_conv(x)
        x = self.dense_block1(x)
        x = self.pool1(x)
        x = self.scf_block1(x)
        x = self.residual_block1(x)
        x = self.pool2(x)
        x = self.dense_block2(x)
        x = self.pool3(x)
        x = self.scf_block2(x)
        x = self.residual_block2(x)
        x = self.pool4(x)
        x = self.dense_block3(x)
        x = self.pool5(x)
        x = self.scf_block3(x)
        x = self.residual_block3(x)
        x = self.pool6(x)
        x = self.global_avg_pool(x)
        return x



if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = XRDNet(num_classes=4, growth_rate=32, dropout_rate=0.5)
    model = model.to(device)
    
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Feature map shape: {features.shape}")