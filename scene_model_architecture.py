import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle

class SceneDataset(Dataset):
    def __init__(self, ref_images, prompts, sketches):
        self.ref_images = torch.FloatTensor(ref_images)  # [N, 12, 64, 64]
        self.prompts = torch.FloatTensor(prompts)        # [N, 39]
        self.sketches = torch.FloatTensor(sketches).unsqueeze(1)  # [N, 1, 256, 256]
        
    def __len__(self):
        return len(self.ref_images)
    
    def __getitem__(self, idx):
        return self.ref_images[idx], self.prompts[idx], self.sketches[idx]

class MultiScaleImageEncoder(nn.Module):
    """Encode multiple reference images with spatial attention"""
    def __init__(self, ref_size=64, feature_dim=256, num_refs=12):
        super(MultiScaleImageEncoder, self).__init__()
        self.num_refs = num_refs
        self.feature_dim = feature_dim
        
        # Shared CNN encoder for all reference images
        self.shared_encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 spatial features
        )
        
        # Global pooling for reference-level features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Reference type embeddings (12 different reference categories)
        self.ref_type_embeddings = nn.Embedding(num_refs, feature_dim)
        
    def forward(self, ref_images):
        # ref_images: [batch_size, num_refs, 64, 64]
        batch_size, num_refs, h, w = ref_images.shape
        
        # Process all reference images through shared encoder
        ref_images_flat = ref_images.view(-1, 1, h, w)  # [batch*num_refs, 1, 64, 64]
        
        # Extract spatial features
        spatial_features = self.shared_encoder(ref_images_flat)  # [batch*num_refs, feature_dim, 8, 8]
        
        # Reshape back to separate references
        spatial_features = spatial_features.view(batch_size, num_refs, self.feature_dim, 8, 8)
        
        # Extract global features for each reference
        global_features = self.global_pool(spatial_features.view(-1, self.feature_dim, 8, 8))
        global_features = global_features.view(batch_size, num_refs, self.feature_dim)
        
        # Add reference type embeddings
        ref_type_ids = torch.arange(num_refs, device=ref_images.device).unsqueeze(0).repeat(batch_size, 1)
        ref_type_embs = self.ref_type_embeddings(ref_type_ids)
        global_features = global_features + ref_type_embs
        
        return spatial_features, global_features

class ScenePromptEncoder(nn.Module):
    """Encode structured scene prompt"""
    def __init__(self, prompt_dim, embed_dim=256):
        super(ScenePromptEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(prompt_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, prompt):
        encoded = self.encoder(prompt)
        return self.norm(encoded)

class HierarchicalAttention(nn.Module):
    """Hierarchical attention: Global scene planning + Local detail extraction"""
    def __init__(self, feature_dim=256, num_heads=8):
        super(HierarchicalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Global attention across references
        self.global_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Spatial attention within references
        self.spatial_query = nn.Linear(feature_dim, feature_dim)
        self.spatial_key = nn.Conv2d(feature_dim, feature_dim, 1)
        self.spatial_value = nn.Conv2d(feature_dim, feature_dim, 1)
        
        # Fusion layers
        self.global_norm = nn.LayerNorm(feature_dim)
        self.spatial_norm = nn.LayerNorm(feature_dim)
        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, prompt_features, spatial_features, global_features):
        # prompt_features: [batch, feature_dim]
        # spatial_features: [batch, num_refs, feature_dim, 8, 8]
        # global_features: [batch, num_refs, feature_dim]
        
        batch_size, num_refs = global_features.shape[:2]
        
        # 1. Global attention across references
        prompt_expanded = prompt_features.unsqueeze(1)  # [batch, 1, feature_dim]
        
        global_attended, global_weights = self.global_attention(
            query=prompt_expanded,
            key=global_features,
            value=global_features
        )
        global_attended = self.global_norm(global_attended.squeeze(1))  # [batch, feature_dim]
        
        # 2. Spatial attention within most relevant references
        # Use global attention weights to select top references
        top_k = min(6, num_refs)  # Focus on top 6 references
        top_ref_indices = torch.topk(global_weights.squeeze(1), top_k, dim=1)[1]  # [batch, top_k]
        
        spatial_attended_list = []
        spatial_attention_maps = []
        
        for batch_idx in range(batch_size):
            batch_spatial_attended = []
            batch_attention_maps = []
            
            for k in range(top_k):
                ref_idx = top_ref_indices[batch_idx, k]
                ref_spatial = spatial_features[batch_idx, ref_idx]  # [feature_dim, 8, 8]
                
                # Spatial attention for this reference
                query = self.spatial_query(prompt_features[batch_idx]).view(1, self.num_heads, self.head_dim)
                key = self.spatial_key(ref_spatial.unsqueeze(0))  # [1, feature_dim, 8, 8]
                value = self.spatial_value(ref_spatial.unsqueeze(0))  # [1, feature_dim, 8, 8]
                
                # Reshape for attention
                key = key.view(1, self.num_heads, self.head_dim, 64)  # [1, heads, head_dim, 64]
                value = value.view(1, self.num_heads, self.head_dim, 64)
                
                # Compute attention
                scores = torch.einsum('bhd,bhdw->bhw', query, key) / (self.head_dim ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)  # [1, heads, 64]
                
                # Apply attention
                attended = torch.einsum('bhw,bhdw->bhd', attn_weights, value)  # [1, heads, head_dim]
                attended = attended.view(1, self.feature_dim)  # [1, feature_dim]
                
                batch_spatial_attended.append(attended.squeeze(0))
                
                # Store attention map for visualization
                attn_map = attn_weights.mean(dim=1).view(8, 8)  # [8, 8]
                batch_attention_maps.append(attn_map)
            
            # Average spatial features from top references
            spatial_attended = torch.stack(batch_spatial_attended).mean(dim=0)  # [feature_dim]
            spatial_attended_list.append(spatial_attended)
            spatial_attention_maps.append(torch.stack(batch_attention_maps))
        
        spatial_attended = torch.stack(spatial_attended_list)  # [batch, feature_dim]
        spatial_attended = self.spatial_norm(spatial_attended)
        
        # 3. Fuse global and spatial attention
        fused_features = torch.cat([global_attended, spatial_attended], dim=1)  # [batch, 2*feature_dim]
        final_features = self.fusion_fc(fused_features)  # [batch, feature_dim]
        
        return final_features, global_weights, spatial_attention_maps

class SceneSketchDecoder(nn.Module):
    """Decode scene features to large sketch canvas"""
    def __init__(self, feature_dim=256, canvas_size=256):
        super(SceneSketchDecoder, self).__init__()
        self.canvas_size = canvas_size
        
        # Progressive upsampling decoder
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8 * 8 * 128),  # Start with 8x8 feature map
            nn.ReLU()
        )
        
        # Decoder network with skip connections
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            
            # Final output layer
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # Project features to spatial map
        x = self.feature_projection(features)
        x = x.view(-1, 128, 8, 8)
        
        # Decode to full size sketch
        sketch = self.decoder(x)
        
        return sketch

class SceneGenerationModel(nn.Module):
    """Complete scene generation model"""
    def __init__(self, prompt_dim, feature_dim=256, num_refs=12):
        super(SceneGenerationModel, self).__init__()
        
        self.image_encoder = MultiScaleImageEncoder(
            ref_size=64, feature_dim=feature_dim, num_refs=num_refs
        )
        self.prompt_encoder = ScenePromptEncoder(
            prompt_dim=prompt_dim, embed_dim=feature_dim
        )
        self.hierarchical_attention = HierarchicalAttention(
            feature_dim=feature_dim, num_heads=8
        )
        self.sketch_decoder = SceneSketchDecoder(
            feature_dim=feature_dim, canvas_size=256
        )
        
    def forward(self, ref_images, prompts):
        # Encode inputs
        spatial_features, global_features = self.image_encoder(ref_images)
        prompt_features = self.prompt_encoder(prompts)
        
        # Hierarchical attention fusion
        fused_features, global_weights, spatial_maps = self.hierarchical_attention(
            prompt_features, spatial_features, global_features
        )
        
        # Generate scene sketch
        scene_sketch = self.sketch_decoder(fused_features)
        
        return scene_sketch, global_weights, spatial_maps

class SceneLoss(nn.Module):
    """Multi-component loss for scene generation"""
    def __init__(self, sketch_weight=1.0, attention_weight=0.1):
        super(SceneLoss, self).__init__()
        self.sketch_weight = sketch_weight
        self.attention_weight = attention_weight
        
        # Weighted MSE for sparse sketches
        self.sketch_loss = nn.MSELoss()
        
    def forward(self, pred_sketch, target_sketch, global_weights=None):
        # Main sketch reconstruction loss with foreground emphasis
        sketch_mse = (pred_sketch - target_sketch) ** 2
        
        # Weight foreground pixels more heavily
        foreground_weight = 20.0
        weights = torch.where(target_sketch > 0.1, foreground_weight, 1.0)
        weighted_sketch_loss = (sketch_mse * weights).mean()
        
        total_loss = self.sketch_weight * weighted_sketch_loss
        
        # Attention regularization (encourage diverse attention)
        if global_weights is not None and self.attention_weight > 0:
            # Encourage attention diversity across references
            attn_entropy = -(global_weights * torch.log(global_weights + 1e-8)).sum(dim=-1).mean()
            attention_loss = -attn_entropy  # Negative because we want high entropy
            total_loss += self.attention_weight * attention_loss
        
        return total_loss, weighted_sketch_loss

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test model architecture
def test_model_architecture():
    print("=== Testing Scene Model Architecture ===")
    
    # Model parameters
    prompt_dim = 39
    feature_dim = 256
    num_refs = 12
    batch_size = 4
    
    # Create model
    model = SceneGenerationModel(
        prompt_dim=prompt_dim,
        feature_dim=feature_dim,
        num_refs=num_refs
    )
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    ref_images = torch.randn(batch_size, num_refs, 64, 64)
    prompts = torch.randn(batch_size, prompt_dim)
    
    print(f"Input shapes:")
    print(f"  Reference images: {ref_images.shape}")
    print(f"  Prompts: {prompts.shape}")
    
    # Forward pass
    with torch.no_grad():
        scene_sketch, global_weights, spatial_maps = model(ref_images, prompts)
    
    print(f"Output shapes:")
    print(f"  Scene sketch: {scene_sketch.shape}")
    print(f"  Global weights: {global_weights.shape}")
    print(f"  Spatial maps: {len(spatial_maps)} batches")
    
    # Test loss
    target_sketch = torch.randn(batch_size, 1, 256, 256)
    loss_fn = SceneLoss()
    total_loss, sketch_loss = loss_fn(scene_sketch, target_sketch, global_weights)
    
    print(f"Loss computation:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Sketch loss: {sketch_loss.item():.4f}")
    
    print("=== Architecture Test Complete ===")

if __name__ == "__main__":
    test_model_architecture()