import numpy as np
import cv2
import random
from typing import Dict, List, Tuple
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from scene_model_architecture import SceneGenerationModel, SceneDataset, SceneLoss, count_parameters
from scene_geometry_generator import SceneGeometryGenerator
from scene_reference_generator import SceneReferenceGenerator


class SceneTrainingDataGenerator:
    def __init__(self, canvas_size=256, ref_size=64, num_references=12):
        self.canvas_size = canvas_size
        self.ref_size = ref_size
        self.num_references = num_references
        
        self.scene_generator = SceneGeometryGenerator(canvas_size, ref_size)
        self.ref_generator = SceneReferenceGenerator(ref_size, canvas_size)
        
        # Scene complexity levels for curriculum learning
        self.complexity_levels = {
            "simple": {"chars": (1, 2), "assets": (1, 3), "interactions": False},
            "medium": {"chars": (2, 3), "assets": (2, 5), "interactions": True},
            "complex": {"chars": (3, 5), "assets": (3, 8), "interactions": True}
        }
    
    def encode_scene_prompt(self, scene_layout: Dict) -> np.ndarray:
        """Encode scene layout into structured prompt vector"""
        encoding = []
        
        # Scene metadata
        num_characters = len(scene_layout["characters"])
        num_assets = len(scene_layout["assets"])
        bg_type = scene_layout["background"]["type"]
        
        # Encode number of characters (0-5, one-hot)
        char_count_encoding = [1 if i == min(num_characters-1, 4) else 0 for i in range(5)]
        encoding.extend(char_count_encoding)
        
        # Encode number of assets (0-8, one-hot)
        asset_count_encoding = [1 if i == min(num_assets-1, 7) else 0 for i in range(8)]
        encoding.extend(asset_count_encoding)
        
        # Encode background type
        bg_types = list(self.scene_generator.background_types.keys())
        bg_encoding = [1 if bg == bg_type else 0 for bg in bg_types]
        encoding.extend(bg_encoding)
        
        # Encode character types present
        char_types = list(self.scene_generator.character_types.keys())
        for char_type in char_types:
            has_char = any(char["type"] == char_type for char in scene_layout["characters"])
            encoding.append(1 if has_char else 0)
        
        # Encode asset types present
        asset_types = list(self.scene_generator.asset_types.keys())
        for asset_type in asset_types:
            has_asset = any(asset["type"] == asset_type for asset in scene_layout["assets"])
            encoding.append(1 if has_asset else 0)
        
        # Encode dominant interaction type
        interactions = ["facing_each_other", "playing_together", "chasing", "following", "standing_apart"]
        if scene_layout["relationships"]:
            dominant_interaction = scene_layout["relationships"][0]["interaction"]
        else:
            dominant_interaction = "standing_apart"
        interaction_encoding = [1 if interaction == dominant_interaction else 0 for interaction in interactions]
        encoding.extend(interaction_encoding)
        
        # Encode scene density (sparse/medium/dense based on total elements)
        total_elements = num_characters + num_assets
        if total_elements <= 3:
            density = "sparse"
        elif total_elements <= 6:
            density = "medium"
        else:
            density = "dense"
        
        density_encoding = [1 if d == density else 0 for d in ["sparse", "medium", "dense"]]
        encoding.extend(density_encoding)
        
        return np.array(encoding, dtype=np.float32)
    
    def generate_scene_references(self, scene_layout: Dict) -> List[np.ndarray]:
        """Generate reference images for scene generation"""
        references = []
        
        # Category 1-2: Character references (heads and bodies)
        unique_char_types = list(set(char["type"] for char in scene_layout["characters"]))
        if len(unique_char_types) >= 1:
            char_type = unique_char_types[0]
            references.append(self.ref_generator.create_noisy_character_reference(char_type, "head"))
            references.append(self.ref_generator.create_noisy_character_reference(char_type, "body"))
        else:
            # Default character references
            references.append(self.ref_generator.create_noisy_character_reference("dog", "head"))
            references.append(self.ref_generator.create_noisy_character_reference("dog", "body"))
        
        # Category 3: Pose/interaction reference
        if scene_layout["relationships"]:
            interaction = scene_layout["relationships"][0]["interaction"]
            references.append(self.ref_generator.create_interaction_reference(interaction))
        else:
            references.append(self.ref_generator.create_pose_reference("standing"))
        
        # Category 4-5: Asset references
        unique_asset_types = list(set(asset["type"] for asset in scene_layout["assets"]))
        for i in range(2):
            if i < len(unique_asset_types):
                asset_type = unique_asset_types[i]
                references.append(self.ref_generator.create_asset_reference(asset_type))
            else:
                # Default asset
                references.append(self.ref_generator.create_asset_reference("tree"))
        
        # Category 6: Background reference
        bg_type = scene_layout["background"]["type"]
        references.append(self.ref_generator.create_background_reference(bg_type))
        
        # Category 7: Layout reference
        num_elements = len(scene_layout["characters"]) + len(scene_layout["assets"])
        layout_type = random.choice(["linear", "circular", "grouped", "scattered"])
        references.append(self.ref_generator.create_layout_reference(num_elements, layout_type))
        
        # Category 8: Secondary character (if exists)
        if len(unique_char_types) >= 2:
            char_type = unique_char_types[1]
            references.append(self.ref_generator.create_noisy_character_reference(char_type, "full_character"))
        else:
            # Add another pose reference
            pose = random.choice(self.scene_generator.poses)
            references.append(self.ref_generator.create_pose_reference(pose))
        
        # Category 9-12: Additional context references
        remaining_slots = self.num_references - len(references)
        
        for i in range(remaining_slots):
            ref_type = random.choice([
                "extra_asset", "extra_character", "extra_layout", "extra_background"
            ])
            
            if ref_type == "extra_asset" and unique_asset_types:
                asset_type = random.choice(unique_asset_types)
                references.append(self.ref_generator.create_asset_reference(asset_type))
            elif ref_type == "extra_character" and unique_char_types:
                char_type = random.choice(unique_char_types)
                focus = random.choice(["head", "body", "full_character"])
                references.append(self.ref_generator.create_noisy_character_reference(char_type, focus))
            elif ref_type == "extra_layout":
                layout_type = random.choice(["linear", "circular", "grouped", "scattered"])
                references.append(self.ref_generator.create_layout_reference(num_elements, layout_type))
            else:
                # Extra background
                references.append(self.ref_generator.create_background_reference(bg_type))
        
        # Ensure we have exactly num_references
        while len(references) < self.num_references:
            references.append(self.ref_generator.create_background_reference("park"))
        
        return references[:self.num_references]
    
    def generate_training_sample(self, complexity: str = "simple") -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict]:
        """Generate one complete training sample"""
        
        # Get complexity parameters
        complexity_params = self.complexity_levels[complexity]
        
        # Generate scene parameters
        num_characters = random.randint(*complexity_params["chars"])
        num_assets = random.randint(*complexity_params["assets"])
        background_type = random.choice(list(self.scene_generator.background_types.keys()))
        
        # Generate scene layout
        scene_layout = self.scene_generator.generate_scene_layout(
            num_characters, num_assets, background_type
        )
        
        # Create scene sketch (target)
        scene_sketch = self.scene_generator.create_scene_sketch(scene_layout)
        
        # Generate reference images
        reference_images = self.generate_scene_references(scene_layout)
        
        # Encode scene prompt
        scene_prompt = self.encode_scene_prompt(scene_layout)
        
        return reference_images, scene_prompt, scene_sketch, scene_layout
    
    def generate_curriculum_dataset(self, samples_per_level: int = 2000, save_path: str = "scene_dataset"):
        """Generate curriculum learning dataset with increasing complexity"""
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        all_ref_images = []
        all_prompts = []
        all_sketches = []
        all_metadata = []
        
        total_samples = 0
        
        # Generate samples for each complexity level
        for complexity in ["simple", "medium", "complex"]:
            print(f"Generating {samples_per_level} {complexity} scenes...")
            
            for i in range(samples_per_level):
                if i % 500 == 0:
                    print(f"  Generated {i}/{samples_per_level} {complexity} samples")
                
                ref_images, prompt, sketch, metadata = self.generate_training_sample(complexity)
                
                ref_images, prompt, sketch, metadata = self.generate_training_sample(complexity)
                
                all_ref_images.append(np.stack(ref_images))  # Stack reference images
                all_prompts.append(prompt)
                all_sketches.append(sketch)
                
                # Add complexity level to metadata
                metadata["complexity"] = complexity
                metadata["sample_id"] = total_samples
                all_metadata.append(metadata)
                
                total_samples += 1
        
        # Convert to numpy arrays
        all_ref_images = np.array(all_ref_images)  # [N, num_refs, ref_size, ref_size]
        all_prompts = np.array(all_prompts)        # [N, prompt_dim]
        all_sketches = np.array(all_sketches)      # [N, canvas_size, canvas_size]
        
        # Save dataset
        np.save(os.path.join(save_path, "reference_images.npy"), all_ref_images)
        np.save(os.path.join(save_path, "scene_prompts.npy"), all_prompts)
        np.save(os.path.join(save_path, "scene_sketches.npy"), all_sketches)
        
        with open(os.path.join(save_path, "scene_metadata.pkl"), "wb") as f:
            pickle.dump(all_metadata, f)
        
        # Save dataset info
        dataset_info = {
            "total_samples": total_samples,
            "samples_per_complexity": samples_per_level,
            "complexity_levels": list(self.complexity_levels.keys()),
            "num_references": self.num_references,
            "canvas_size": self.canvas_size,
            "ref_size": self.ref_size,
            "prompt_dim": all_prompts.shape[1]
        }
        
        with open(os.path.join(save_path, "dataset_info.json"), "w") as f:
            import json
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Total samples: {total_samples}")
        print(f"Reference images shape: {all_ref_images.shape}")
        print(f"Scene prompts shape: {all_prompts.shape}")
        print(f"Scene sketches shape: {all_sketches.shape}")
        print(f"Dataset saved to: {save_path}")
        
        return all_ref_images, all_prompts, all_sketches, all_metadata
    
    def generate_test_visualizations(self, num_samples: int = 5, save_path: str = "scene_visualizations"):
        """Generate test samples for visualization"""
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for complexity in ["simple", "medium", "complex"]:
            for sample_idx in range(num_samples):
                # Generate sample
                ref_images, prompt, sketch, metadata = self.generate_training_sample(complexity)
                
                # Create visualization
                self._create_sample_visualization(
                    ref_images, prompt, sketch, metadata, 
                    f"{save_path}/{complexity}_sample_{sample_idx}.png"
                )
        
        print(f"Test visualizations saved to {save_path}")
    
    def _create_sample_visualization(self, ref_images: List[np.ndarray], prompt: np.ndarray, 
                                   sketch: np.ndarray, metadata: Dict, save_path: str):
        """Create visualization of a training sample"""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            
            # Show reference images (3x4 grid)
            ref_titles = [
                "Char Head", "Char Body", "Interaction", "Asset 1", 
                "Asset 2", "Background", "Layout", "Char 2",
                "Extra 1", "Extra 2", "Extra 3", "Extra 4"
            ]
            
            for i in range(min(12, len(ref_images))):
                row = i // 4
                col = i % 4
                axes[row, col].imshow(ref_images[i], cmap='gray')
                axes[row, col].set_title(ref_titles[i], fontsize=10)
                axes[row, col].axis('off')
            
            # Show target sketch
            axes[3, 0].imshow(sketch, cmap='gray')
            axes[3, 0].set_title('Target Scene Sketch', fontsize=12)
            axes[3, 0].axis('off')
            
            # Show scene metadata
            scene_info = f"""Scene: {metadata.get('complexity', 'Unknown')}
Characters: {len(metadata.get('characters', []))}
Assets: {len(metadata.get('assets', []))}
Background: {metadata.get('background', {}).get('type', 'Unknown')}
Relationships: {len(metadata.get('relationships', []))}"""
            
            axes[3, 1].text(0.1, 0.5, scene_info, fontsize=10, verticalalignment='center')
            axes[3, 1].set_title('Scene Info', fontsize=12)
            axes[3, 1].axis('off')
            
            # Show prompt encoding info
            prompt_info = f"""Prompt Dimensions: {len(prompt)}
Char Count Encoding: {prompt[:5]}
Asset Count Encoding: {prompt[5:13]}
Background Encoding: {prompt[13:18]}
Interaction Encoding: {prompt[-8:-3]}"""
            
            axes[3, 2].text(0.1, 0.5, prompt_info, fontsize=8, verticalalignment='center', family='monospace')
            axes[3, 2].set_title('Prompt Encoding', fontsize=12)
            axes[3, 2].axis('off')
            
            # Hide empty subplot
            axes[3, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Matplotlib not available - skipping visualization")
    
    def analyze_dataset_statistics(self, dataset_path: str):
        """Analyze generated dataset statistics"""
        
        # Load dataset
        ref_images = np.load(os.path.join(dataset_path, "reference_images.npy"))
        prompts = np.load(os.path.join(dataset_path, "scene_prompts.npy"))
        sketches = np.load(os.path.join(dataset_path, "scene_sketches.npy"))
        
        with open(os.path.join(dataset_path, "scene_metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        
        print("=== Dataset Statistics ===")
        print(f"Total samples: {len(ref_images)}")
        print(f"Reference images shape: {ref_images.shape}")
        print(f"Prompts shape: {prompts.shape}")
        print(f"Sketches shape: {sketches.shape}")
        
        # Analyze complexity distribution
        complexity_counts = {}
        for meta in metadata:
            complexity = meta.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        print(f"\nComplexity distribution:")
        for complexity, count in complexity_counts.items():
            print(f"  {complexity}: {count} samples")
        
        # Analyze character and asset distributions
        char_counts = {}
        asset_counts = {}
        bg_counts = {}
        
        for meta in metadata:
            # Character types
            for char in meta.get("characters", []):
                char_type = char["type"]
                char_counts[char_type] = char_counts.get(char_type, 0) + 1
            
            # Asset types
            for asset in meta.get("assets", []):
                asset_type = asset["type"]
                asset_counts[asset_type] = asset_counts.get(asset_type, 0) + 1
            
            # Background types
            bg_type = meta.get("background", {}).get("type", "unknown")
            bg_counts[bg_type] = bg_counts.get(bg_type, 0) + 1
        
        print(f"\nCharacter type distribution:")
        for char_type, count in sorted(char_counts.items()):
            print(f"  {char_type}: {count}")
        
        print(f"\nAsset type distribution:")
        for asset_type, count in sorted(asset_counts.items()):
            print(f"  {asset_type}: {count}")
        
        print(f"\nBackground type distribution:")
        for bg_type, count in sorted(bg_counts.items()):
            print(f"  {bg_type}: {count}")
        
        # Analyze prompt encoding statistics
        print(f"\nPrompt encoding statistics:")
        print(f"  Min value: {prompts.min():.3f}")
        print(f"  Max value: {prompts.max():.3f}")
        print(f"  Mean value: {prompts.mean():.3f}")
        print(f"  Non-zero rate: {(prompts > 0).mean():.3f}")
        
        # Analyze sketch statistics
        print(f"\nSketch statistics:")
        print(f"  Min value: {sketches.min():.3f}")
        print(f"  Max value: {sketches.max():.3f}")
        print(f"  Mean value: {sketches.mean():.3f}")
        print(f"  Non-zero pixels per sketch: {(sketches > 0).sum(axis=(1,2)).mean():.1f}")
        
        return {
            "complexity_counts": complexity_counts,
            "char_counts": char_counts,
            "asset_counts": asset_counts,
            "bg_counts": bg_counts
        }

# Test the complete pipeline
def data_main():
    print("=== Scene Training Pipeline Test ===")
    
    # Initialize generator
    generator = SceneTrainingDataGenerator(
        canvas_size=256, 
        ref_size=64, 
        num_references=12
    )
    
    # Test single sample generation
    print("\n1. Testing single sample generation...")
    for complexity in ["simple", "medium", "complex"]:
        print(f"\nGenerating {complexity} scene...")
        ref_images, prompt, sketch, metadata = generator.generate_training_sample(complexity)
        
        print(f"  Reference images: {len(ref_images)} x {ref_images[0].shape}")
        print(f"  Prompt shape: {prompt.shape}")
        print(f"  Sketch shape: {sketch.shape}")
        print(f"  Characters: {len(metadata['characters'])}")
        print(f"  Assets: {len(metadata['assets'])}")
        print(f"  Background: {metadata['background']['type']}")
    
    # Generate test visualizations
    print("\n2. Generating test visualizations...")
    generator.generate_test_visualizations(num_samples=2)
    
    # Generate small dataset for testing
    print("\n3. Generating small test dataset...")
    ref_images, prompts, sketches, metadata = generator.generate_curriculum_dataset(
        samples_per_level=100,  # Small dataset for testing
        save_path="test_scene_dataset"
    )
    
    # Analyze dataset
    print("\n4. Analyzing dataset...")
    stats = generator.analyze_dataset_statistics("test_scene_dataset")
    
    print("\n=== Pipeline Test Complete ===")
    print("Ready for model training!")


class SceneModelTrainer:
    def __init__(self, model, device='cuda', learning_rate=0.001):  # Increased LR from 0.0001
        self.model = model.to(device)
        self.device = device
        
        # Simplified optimizer with higher learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4  # Reduced weight decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5  # Reduce LR every 20 epochs
        )
        
        # Fixed loss function
        self.criterion = SceneLoss(sketch_weight=1.0, attention_weight=0.01)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_sketch_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (ref_images, prompts, target_sketches) in enumerate(train_loader):
            ref_images = ref_images.to(self.device)
            prompts = prompts.to(self.device)
            target_sketches = target_sketches.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_sketches, global_weights, spatial_maps = self.model(ref_images, prompts)
            
            # Compute loss
            total_loss_batch, sketch_loss_batch = self.criterion(
                pred_sketches, target_sketches, global_weights
            )
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_sketch_loss += sketch_loss_batch.item()
            
            # Debug output for first few batches
            if epoch == 0 and batch_idx < 3:
                print(f"  Batch {batch_idx}: Loss = {total_loss_batch.item():.4f}")
                print(f"    Pred sketch range: [{pred_sketches.min():.3f}, {pred_sketches.max():.3f}]")
                print(f"    Target sketch range: [{target_sketches.min():.3f}, {target_sketches.max():.3f}]")
                print(f"    Global attention shape: {global_weights.shape}")
            
            # Progress update
            if batch_idx % (num_batches // 5) == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {total_loss_batch.item():.4f}")
        
        avg_loss = total_loss / num_batches
        avg_sketch_loss = total_sketch_loss / num_batches
        
        return avg_loss, avg_sketch_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_sketch_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for ref_images, prompts, target_sketches in val_loader:
                ref_images = ref_images.to(self.device)
                prompts = prompts.to(self.device)
                target_sketches = target_sketches.to(self.device)
                
                # Forward pass
                pred_sketches, global_weights, spatial_maps = self.model(ref_images, prompts)
                
                # Compute loss
                total_loss_batch, sketch_loss_batch = self.criterion(
                    pred_sketches, target_sketches, global_weights
                )
                
                total_loss += total_loss_batch.item()
                total_sketch_loss += sketch_loss_batch.item()
        
        avg_loss = total_loss / num_batches
        avg_sketch_loss = total_sketch_loss / num_batches
        
        return avg_loss, avg_sketch_loss
    
    def train(self, train_loader, val_loader, num_epochs=80, save_path="scene_model_checkpoints"):
        """Full training loop"""
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_sketch_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_sketch_loss = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()  # Changed from scheduler.step(val_loss)
            
            # Save losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"  Train Loss: {train_loss:.4f} (Sketch: {train_sketch_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (Sketch: {val_sketch_loss:.4f})")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'best_val_loss': self.best_val_loss
                }, os.path.join(save_path, 'best_scene_model.pth'))
                print(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, os.path.join(save_path, f'scene_model_epoch_{epoch+1}.pth'))
                
                # Plot training curves
                self.plot_training_curves(os.path.join(save_path, f'training_curves_epoch_{epoch+1}.png'))
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def plot_training_curves(self, save_path):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Plot recent epochs only for better detail
        recent_epochs = min(20, len(self.train_losses))
        recent_train = self.train_losses[-recent_epochs:]
        recent_val = self.val_losses[-recent_epochs:]
        
        plt.plot(range(len(self.train_losses)-recent_epochs, len(self.train_losses)), recent_train, label='Train Loss (Recent)')
        plt.plot(range(len(self.val_losses)-recent_epochs, len(self.val_losses)), recent_val, label='Val Loss (Recent)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def visualize_scene_predictions(model, val_loader, device, num_samples=3, save_path="scene_predictions"):
    """Visualize model predictions on validation data"""
    model.eval()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with torch.no_grad():
        ref_images, prompts, target_sketches = next(iter(val_loader))
        ref_images = ref_images.to(device)
        prompts = prompts.to(device)
        target_sketches = target_sketches.to(device)
        
        pred_sketches, global_weights, spatial_maps = model(ref_images, prompts)
        
        for sample_idx in range(min(num_samples, len(ref_images))):
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            
            # Show reference images (top 3 rows)
            ref_titles = [
                "Char Head", "Char Body", "Interaction", "Asset 1",
                "Asset 2", "Background", "Layout", "Char 2", 
                "Extra 1", "Extra 2", "Extra 3", "Extra 4"
            ]
            
            for ref_idx in range(12):
                row = ref_idx // 4
                col = ref_idx % 4
                axes[row, col].imshow(ref_images[sample_idx, ref_idx].cpu().numpy(), cmap='gray')
                
                # Show attention weight in title
                if ref_idx < global_weights.shape[2]:
                    weight = global_weights[sample_idx, 0, ref_idx].cpu().item()
                    axes[row, col].set_title(f'{ref_titles[ref_idx]}\n(Attn: {weight:.3f})', fontsize=10)
                else:
                    axes[row, col].set_title(ref_titles[ref_idx], fontsize=10)
                
                axes[row, col].axis('off')
            
            # Show predictions vs targets (bottom row)
            axes[3, 0].imshow(target_sketches[sample_idx, 0].cpu().numpy(), cmap='gray')
            axes[3, 0].set_title('Target Scene Sketch', fontsize=12)
            axes[3, 0].axis('off')
            
            axes[3, 1].imshow(pred_sketches[sample_idx, 0].cpu().numpy(), cmap='gray')
            axes[3, 1].set_title('Predicted Scene Sketch', fontsize=12)
            axes[3, 1].axis('off')
            
            # Show difference map
            diff = torch.abs(target_sketches[sample_idx, 0] - pred_sketches[sample_idx, 0])
            axes[3, 2].imshow(diff.cpu().numpy(), cmap='hot')
            axes[3, 2].set_title('Difference Map', fontsize=12)
            axes[3, 2].axis('off')
            
            # Show global attention weights
            weights = global_weights[sample_idx, 0].cpu().numpy()
            axes[3, 3].bar(range(len(weights)), weights)
            axes[3, 3].set_title('Global Attention Weights', fontsize=12)
            axes[3, 3].set_xlabel('Reference Image')
            axes[3, 3].set_ylabel('Attention Weight')
            axes[3, 3].set_xticks(range(0, len(weights), 2))
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/scene_prediction_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Scene predictions saved to {save_path}")

def load_scene_dataset(dataset_path):
    """Load scene dataset from saved files"""
    print(f"Loading dataset from {dataset_path}...")
    
    # Load arrays
    ref_images = np.load(os.path.join(dataset_path, "reference_images.npy"))
    prompts = np.load(os.path.join(dataset_path, "scene_prompts.npy"))
    sketches = np.load(os.path.join(dataset_path, "scene_sketches.npy"))
    
    # Load metadata
    with open(os.path.join(dataset_path, "scene_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    # Load dataset info
    with open(os.path.join(dataset_path, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    
    print(f"Dataset loaded:")
    print(f"  Reference images: {ref_images.shape}")
    print(f"  Prompts: {prompts.shape}")
    print(f"  Sketches: {sketches.shape}")
    print(f"  Total samples: {len(metadata)}")
    
    return ref_images, prompts, sketches, metadata, dataset_info

def create_curriculum_dataloaders(ref_images, prompts, sketches, metadata, batch_size=8, val_split=0.2):
    """Create data loaders with curriculum learning organization"""
    
    # Organize by complexity
    simple_indices = []
    medium_indices = []
    complex_indices = []
    
    for i, meta in enumerate(metadata):
        complexity = meta.get("complexity", "simple")
        if complexity == "simple":
            simple_indices.append(i)
        elif complexity == "medium":
            medium_indices.append(i)
        else:
            complex_indices.append(i)
    
    print(f"Dataset split by complexity:")
    print(f"  Simple: {len(simple_indices)} samples")
    print(f"  Medium: {len(medium_indices)} samples")
    print(f"  Complex: {len(complex_indices)} samples")
    
    # For initial training, we'll use all complexities mixed
    # In practice, you might want to train curriculum: simple -> medium -> complex
    all_indices = simple_indices + medium_indices + complex_indices
    np.random.shuffle(all_indices)
    
    # Train/validation split
    n_val = int(len(all_indices) * val_split)
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]
    
    print(f"Train/Val split:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Validation: {len(val_indices)} samples")
    
    # Create datasets
    train_dataset = SceneDataset(
        ref_images[train_indices],
        prompts[train_indices],
        sketches[train_indices]
    )
    
    val_dataset = SceneDataset(
        ref_images[val_indices],
        prompts[val_indices],
        sketches[val_indices]
    )
    
    # Create data loaders (single-threaded to avoid multiprocessing issues)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader

def main():
    print("=== Scene Model Training ===")
    
    # Configuration
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    dataset_path = "test_scene_dataset"  # Use the test dataset we generated
    batch_size = 6  # Slightly larger batch size for better gradient estimates
    learning_rate = 0.001  # Much higher learning rate for faster initial learning
    num_epochs = 80  # Enough epochs to see convergence
    
    # Load dataset
    ref_images, prompts, sketches, metadata, dataset_info = load_scene_dataset(dataset_path)
    
    # Create data loaders
    train_loader, val_loader = create_curriculum_dataloaders(
        ref_images, prompts, sketches, metadata, 
        batch_size=batch_size, val_split=0.2
    )
    
    # Create model
    prompt_dim = dataset_info["prompt_dim"]
    num_refs = dataset_info["num_references"]
    
    model = SceneGenerationModel(
        prompt_dim=prompt_dim,
        feature_dim=256,
        num_refs=num_refs
    )
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Create trainer
    trainer = SceneModelTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Test model with one batch first
    print("\nTesting model with one batch...")
    test_ref, test_prompts, test_sketches = next(iter(train_loader))
    test_ref = test_ref.to(device)
    test_prompts = test_prompts.to(device)
    test_sketches = test_sketches.to(device)
    
    with torch.no_grad():
        pred_sketches, global_weights, spatial_maps = model(test_ref, test_prompts)
        print(f"Test forward pass successful!")
        print(f"  Input: {test_ref.shape} -> Output: {pred_sketches.shape}")
        print(f"  Global attention: {global_weights.shape}")
        print(f"  Spatial maps: {len(spatial_maps)} batches")
    
    # Train model
    print(f"\nStarting training...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path="scene_model_checkpoints"
    )
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load("scene_model_checkpoints/best_scene_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate predictions
    print("Generating prediction visualizations...")
    visualize_scene_predictions(
        model=model,
        val_loader=val_loader,
        device=device,
        num_samples=5,
        save_path="scene_predictions"
    )
    
    # Final training curves
    trainer.plot_training_curves("final_training_curves.png")
    
    print("=== Training Complete ===")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print("Check 'scene_predictions/' for generated scene sketches!")

def test_curriculum_training():
    """Test curriculum training approach: simple -> medium -> complex"""
    print("=== Testing Curriculum Training ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = "test_scene_dataset"
    
    # Load dataset
    ref_images, prompts, sketches, metadata, dataset_info = load_scene_dataset(dataset_path)
    
    # Separate by complexity
    complexity_data = {"simple": [], "medium": [], "complex": []}
    
    for i, meta in enumerate(metadata):
        complexity = meta.get("complexity", "simple")
        complexity_data[complexity].append(i)
    
    # Create model
    model = SceneGenerationModel(
        prompt_dim=dataset_info["prompt_dim"],
        feature_dim=256,
        num_refs=dataset_info["num_references"]
    )
    
    trainer = SceneModelTrainer(model=model, device=device, learning_rate=0.0002)
    
    # Stage 1: Train on simple scenes
    print("\n=== Stage 1: Training on Simple Scenes ===")
    simple_indices = complexity_data["simple"]
    train_split = int(0.8 * len(simple_indices))
    
    simple_train_dataset = SceneDataset(
        ref_images[simple_indices[:train_split]],
        prompts[simple_indices[:train_split]],
        sketches[simple_indices[:train_split]]
    )
    simple_val_dataset = SceneDataset(
        ref_images[simple_indices[train_split:]],
        prompts[simple_indices[train_split:]],
        sketches[simple_indices[train_split:]]
    )
    
    simple_train_loader = DataLoader(simple_train_dataset, batch_size=4, shuffle=True, num_workers=0)
    simple_val_loader = DataLoader(simple_val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Train on simple for fewer epochs
    trainer.train(simple_train_loader, simple_val_loader, num_epochs=20, save_path="curriculum_stage1")
    
    # Stage 2: Fine-tune on medium complexity
    print("\n=== Stage 2: Fine-tuning on Medium Scenes ===")
    # Reduce learning rate for fine-tuning
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] *= 0.5
    
    medium_indices = complexity_data["medium"]
    train_split = int(0.8 * len(medium_indices))
    
    medium_train_dataset = SceneDataset(
        ref_images[medium_indices[:train_split]],
        prompts[medium_indices[:train_split]],
        sketches[medium_indices[:train_split]]
    )
    medium_val_dataset = SceneDataset(
        ref_images[medium_indices[train_split:]],
        prompts[medium_indices[train_split:]],
        sketches[medium_indices[train_split:]]
    )
    
    medium_train_loader = DataLoader(medium_train_dataset, batch_size=4, shuffle=True, num_workers=0)
    medium_val_loader = DataLoader(medium_val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    trainer.train(medium_train_loader, medium_val_loader, num_epochs=15, save_path="curriculum_stage2")
    
    # Stage 3: Fine-tune on complex scenes
    print("\n=== Stage 3: Fine-tuning on Complex Scenes ===")
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] *= 0.5
    
    complex_indices = complexity_data["complex"]
    train_split = int(0.8 * len(complex_indices))
    
    complex_train_dataset = SceneDataset(
        ref_images[complex_indices[:train_split]],
        prompts[complex_indices[:train_split]],
        sketches[complex_indices[:train_split]]
    )
    complex_val_dataset = SceneDataset(
        ref_images[complex_indices[train_split:]],
        prompts[complex_indices[train_split:]],
        sketches[complex_indices[train_split:]]
    )
    
    complex_train_loader = DataLoader(complex_train_dataset, batch_size=4, shuffle=True, num_workers=0)
    complex_val_loader = DataLoader(complex_val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    trainer.train(complex_train_loader, complex_val_loader, num_epochs=20, save_path="curriculum_stage3")
    
    print("=== Curriculum Training Complete ===")

if __name__ == "__main__":
    # Run main training
    data_main()
    main()
    
    # Optionally test curriculum approach
    # test_curriculum_training()