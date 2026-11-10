import numpy as np
import cv2
import random
from typing import Dict, List, Tuple
import pickle
import os
from scene_geometry_generator import SceneGeometryGenerator

class SceneReferenceGenerator:
    def __init__(self, ref_size=64, canvas_size=256):
        self.ref_size = ref_size
        self.canvas_size = canvas_size
        self.scene_generator = SceneGeometryGenerator(canvas_size, ref_size)
        
        # Reference categories for scene generation
        self.reference_categories = [
            "character_heads",      # Different character head types
            "character_bodies",     # Different character body types  
            "character_poses",      # Different poses and interactions
            "background_elements",  # Trees, houses, etc.
            "small_objects",        # Balls, flowers, rocks
            "scene_layouts",        # Overall composition examples
            "spatial_relationships", # How characters interact spatially
            "background_settings"   # Different environment types
        ]
    
    def create_noisy_character_reference(self, char_type: str, focus: str) -> np.ndarray:
        """Create noisy reference focusing on specific character aspect"""
        ref_img = np.zeros((self.ref_size, self.ref_size), dtype=np.float32)
        
        # Add background noise
        ref_img = self._add_complex_background_noise(ref_img)
        
        # Create the target character element
        center_x = self.ref_size // 2 + random.randint(-3, 3)
        center_y = self.ref_size // 2 + random.randint(-3, 3)
        
        char_def = self.scene_generator.character_types[char_type]
        base_size = random.randint(8, 12)
        
        if focus == "head":
            self._draw_character_head(ref_img, center_x, center_y, char_def["head"], base_size)
        elif focus == "body":
            self._draw_character_body(ref_img, center_x, center_y, char_def["body"], base_size)
        elif focus == "full_character":
            # Draw complete character but with some variation
            self._draw_full_character_reference(ref_img, center_x, center_y, char_type, base_size)
        
        # Add distractor objects around the target
        ref_img = self._add_character_distractors(ref_img, avoid_center=True)
        
        return ref_img
    
    def create_pose_reference(self, pose: str, char_type: str = None) -> np.ndarray:
        """Create reference showing specific pose/interaction"""
        ref_img = np.zeros((self.ref_size, self.ref_size), dtype=np.float32)
        
        # Add background noise
        ref_img = self._add_pose_background_noise(ref_img)
        
        # Draw pose example
        center_x, center_y = self.ref_size // 2, self.ref_size // 2
        base_size = 10
        
        if char_type:
            # Draw specific character in pose
            self._draw_character_in_pose(ref_img, center_x, center_y, char_type, pose, base_size)
        else:
            # Draw abstract pose representation
            self._draw_abstract_pose(ref_img, center_x, center_y, pose, base_size)
        
        # Add pose-related distractors
        ref_img = self._add_pose_distractors(ref_img)
        
        return ref_img
    
    def create_asset_reference(self, asset_type: str) -> np.ndarray:
        """Create reference for scene assets/objects"""
        ref_img = np.zeros((self.ref_size, self.ref_size), dtype=np.float32)
        
        # Add environmental noise
        ref_img = self._add_environmental_noise(ref_img)
        
        # Draw the target asset
        center_x = self.ref_size // 2 + random.randint(-2, 2)
        center_y = self.ref_size // 2 + random.randint(-2, 2)
        
        # Create temporary asset dict for drawing
        asset = {
            "type": asset_type,
            "position": (center_x, center_y),
            "region": "midground",
            "size": self.scene_generator.asset_types[asset_type]["size"]
        }
        
        self.scene_generator.draw_asset(ref_img, asset, color=1.0)
        
        # Add related but different objects as distractors
        ref_img = self._add_asset_distractors(ref_img, asset_type)
        
        return ref_img
    
    def create_background_reference(self, bg_type: str) -> np.ndarray:
        """Create reference for background/environment"""
        ref_img = np.zeros((self.ref_size, self.ref_size), dtype=np.float32)
        
        # Create miniature version of background
        bg_def = self.scene_generator.background_types[bg_type]
        
        # Draw ground line
        ground_y = int(self.ref_size * 0.7)
        cv2.line(ref_img, (0, ground_y), (self.ref_size, ground_y), 0.8, 1)
        
        # Add background-specific elements
        if bg_type == "park":
            self._draw_park_elements(ref_img, ground_y)
        elif bg_type == "city":
            self._draw_city_elements(ref_img, ground_y)
        elif bg_type == "beach":
            self._draw_beach_elements(ref_img, ground_y)
        elif bg_type == "space":
            self._draw_space_elements(ref_img)
        elif bg_type == "forest":
            self._draw_forest_elements(ref_img, ground_y)
        
        # Add noise to make it more realistic
        ref_img = self._add_background_texture_noise(ref_img)
        
        return ref_img
    
    def create_layout_reference(self, num_elements: int, layout_type: str) -> np.ndarray:
        """Create reference showing spatial layout patterns"""
        ref_img = np.zeros((self.ref_size, self.ref_size), dtype=np.float32)
        
        # Add subtle background
        ref_img = self._add_layout_background(ref_img)
        
        # Draw layout pattern
        if layout_type == "linear":
            self._draw_linear_layout(ref_img, num_elements)
        elif layout_type == "circular":
            self._draw_circular_layout(ref_img, num_elements)
        elif layout_type == "grouped":
            self._draw_grouped_layout(ref_img, num_elements)
        elif layout_type == "scattered":
            self._draw_scattered_layout(ref_img, num_elements)
        
        return ref_img
    
    def create_interaction_reference(self, interaction: str) -> np.ndarray:
        """Create reference showing character interactions"""
        ref_img = np.zeros((self.ref_size, self.ref_size), dtype=np.float32)
        
        # Add background noise
        ref_img = self._add_interaction_background(ref_img)
        
        # Draw interaction example with simple shapes
        if interaction == "facing_each_other":
            self._draw_facing_interaction(ref_img)
        elif interaction == "playing_together":
            self._draw_playing_interaction(ref_img)
        elif interaction == "chasing":
            self._draw_chasing_interaction(ref_img)
        elif interaction == "following":
            self._draw_following_interaction(ref_img)
        elif interaction == "standing_apart":
            self._draw_apart_interaction(ref_img)
        
        return ref_img
    
    # Helper methods for drawing specific elements
    def _draw_character_head(self, img: np.ndarray, x: int, y: int, head_type: str, size: int):
        """Draw character head with some detail"""
        if head_type == "circle":
            cv2.circle(img, (x, y), size, 1.0, 2)
            # Add facial features
            cv2.circle(img, (x-size//3, y-size//4), 1, 0.8, -1)  # Eye
            cv2.circle(img, (x+size//3, y-size//4), 1, 0.8, -1)  # Eye
        elif head_type == "square":
            cv2.rectangle(img, (x-size, y-size), (x+size, y+size), 1.0, 2)
        elif head_type == "oval":
            cv2.ellipse(img, (x, y), (size, int(size*0.7)), 0, 0, 360, 1.0, 2)
    
    def _draw_character_body(self, img: np.ndarray, x: int, y: int, body_type: str, size: int):
        """Draw character body"""
        if body_type == "oval":
            cv2.ellipse(img, (x, y), (size//2, size), 0, 0, 360, 1.0, 2)
        elif body_type == "rectangle":
            cv2.rectangle(img, (x-size//2, y-size), (x+size//2, y+size), 1.0, 2)
        elif body_type == "triangle":
            pts = np.array([[x, y-size], [x-size, y+size], [x+size, y+size]], np.int32)
            cv2.polylines(img, [pts], True, 1.0, 2)
    
    def _draw_full_character_reference(self, img: np.ndarray, x: int, y: int, char_type: str, size: int):
        """Draw complete character as reference"""
        char_def = self.scene_generator.character_types[char_type]
        
        # Head
        head_y = y - size
        self._draw_character_head(img, x, head_y, char_def["head"], size//2)
        
        # Body
        self._draw_character_body(img, x, y, char_def["body"], size//2)
        
        # Simple limbs
        arm_len = size//3
        cv2.line(img, (x, y), (x-arm_len, y+arm_len), 1.0, 1)
        cv2.line(img, (x, y), (x+arm_len, y+arm_len), 1.0, 1)
    
    def _draw_character_in_pose(self, img: np.ndarray, x: int, y: int, char_type: str, pose: str, size: int):
        """Draw character in specific pose"""
        # Draw basic character
        self._draw_full_character_reference(img, x, y, char_type, size)
        
        # Add pose-specific elements
        if pose == "jumping":
            # Add motion lines
            cv2.line(img, (x-size, y+size), (x-size//2, y+size//2), 0.6, 1)
            cv2.line(img, (x+size, y+size), (x+size//2, y+size//2), 0.6, 1)
        elif pose == "running":
            # Add speed lines
            for i in range(3):
                cv2.line(img, (x-size-i*2, y-i), (x-size//2-i*2, y-i), 0.5, 1)
    
    def _draw_abstract_pose(self, img: np.ndarray, x: int, y: int, pose: str, size: int):
        """Draw abstract representation of pose"""
        if pose == "standing":
            cv2.line(img, (x, y-size), (x, y+size), 1.0, 2)  # Vertical line
        elif pose == "sitting":
            cv2.line(img, (x-size, y), (x+size, y), 1.0, 2)  # Horizontal line
        elif pose == "walking":
            # Diagonal stride pattern
            cv2.line(img, (x-size//2, y), (x+size//2, y+size//2), 1.0, 2)
        elif pose == "running":
            # Dynamic diagonal
            cv2.line(img, (x-size, y-size//2), (x+size, y+size//2), 1.0, 2)
        elif pose == "jumping":
            # Arc motion
            pts = np.array([[x-size, y], [x, y-size], [x+size, y]], np.int32)
            cv2.polylines(img, [pts], False, 1.0, 2)
    
    # Background element drawing methods
    def _draw_park_elements(self, img: np.ndarray, ground_y: int):
        """Draw park-specific elements"""
        # Small trees
        for i in range(2):
            x = random.randint(5, self.ref_size-5)
            tree_h = random.randint(8, 12)
            # Trunk
            cv2.line(img, (x, ground_y), (x, ground_y-tree_h//2), 0.6, 1)
            # Leaves
            cv2.circle(img, (x, ground_y-tree_h), tree_h//3, 0.6, 1)
    
    def _draw_city_elements(self, img: np.ndarray, ground_y: int):
        """Draw city-specific elements"""
        # Buildings
        for i in range(3):
            x = i * (self.ref_size // 3) + 5
            building_h = random.randint(10, 15)
            cv2.rectangle(img, (x, ground_y-building_h), (x+8, ground_y), 0.6, 1)
    
    def _draw_beach_elements(self, img: np.ndarray, ground_y: int):
        """Draw beach-specific elements"""
        # Waves
        wave_y = ground_y + 3
        for x in range(0, self.ref_size, 8):
            cv2.line(img, (x, wave_y), (x+4, wave_y-2), 0.5, 1)
            cv2.line(img, (x+4, wave_y-2), (x+8, wave_y), 0.5, 1)
    
    def _draw_space_elements(self, img: np.ndarray):
        """Draw space-specific elements"""
        # Stars
        for i in range(10):
            x, y = random.randint(0, self.ref_size), random.randint(0, self.ref_size//2)
            cv2.circle(img, (x, y), 1, 0.8, -1)
        # Planet
        cv2.circle(img, (self.ref_size//4, self.ref_size//4), 6, 0.7, 2)
    
    def _draw_forest_elements(self, img: np.ndarray, ground_y: int):
        """Draw forest-specific elements"""
        # Dense trees
        for i in range(4):
            x = random.randint(5, self.ref_size-5)
            tree_h = random.randint(10, 16)
            cv2.line(img, (x, ground_y), (x, ground_y-tree_h), 0.6, 2)
    
    # Layout drawing methods
    def _draw_linear_layout(self, img: np.ndarray, num_elements: int):
        """Draw elements in a line"""
        spacing = self.ref_size // (num_elements + 1)
        y = self.ref_size // 2
        for i in range(num_elements):
            x = spacing * (i + 1)
            cv2.circle(img, (x, y), 3, 1.0, -1)
    
    def _draw_circular_layout(self, img: np.ndarray, num_elements: int):
        """Draw elements in a circle"""
        center_x, center_y = self.ref_size // 2, self.ref_size // 2
        radius = self.ref_size // 3
        for i in range(num_elements):
            angle = 2 * np.pi * i / num_elements
            x = center_x + int(radius * np.cos(angle))
            y = center_y + int(radius * np.sin(angle))
            cv2.circle(img, (x, y), 2, 1.0, -1)
    
    def _draw_grouped_layout(self, img: np.ndarray, num_elements: int):
        """Draw elements in clusters"""
        # Create 2 groups
        group1_center = (self.ref_size//3, self.ref_size//2)
        group2_center = (2*self.ref_size//3, self.ref_size//2)
        
        for i in range(num_elements):
            if i < num_elements//2:
                center = group1_center
            else:
                center = group2_center
            
            x = center[0] + random.randint(-5, 5)
            y = center[1] + random.randint(-5, 5)
            cv2.circle(img, (x, y), 2, 1.0, -1)
    
    def _draw_scattered_layout(self, img: np.ndarray, num_elements: int):
        """Draw elements randomly scattered"""
        for i in range(num_elements):
            x = random.randint(5, self.ref_size-5)
            y = random.randint(5, self.ref_size-5)
            cv2.circle(img, (x, y), 2, 1.0, -1)
    
    # Interaction drawing methods
    def _draw_facing_interaction(self, img: np.ndarray):
        """Draw two elements facing each other"""
        y = self.ref_size // 2
        # Left character
        cv2.circle(img, (self.ref_size//3, y), 4, 1.0, 2)
        cv2.line(img, (self.ref_size//3+2, y), (self.ref_size//3+6, y), 1.0, 1)  # Arrow pointing right
        # Right character  
        cv2.circle(img, (2*self.ref_size//3, y), 4, 1.0, 2)
        cv2.line(img, (2*self.ref_size//3-2, y), (2*self.ref_size//3-6, y), 1.0, 1)  # Arrow pointing left
    
    def _draw_playing_interaction(self, img: np.ndarray):
        """Draw playing interaction"""
        # Two characters with object between
        cv2.circle(img, (self.ref_size//4, self.ref_size//2), 3, 1.0, 2)
        cv2.circle(img, (3*self.ref_size//4, self.ref_size//2), 3, 1.0, 2)
        cv2.circle(img, (self.ref_size//2, self.ref_size//2), 2, 0.8, -1)  # Ball
    
    def _draw_chasing_interaction(self, img: np.ndarray):
        """Draw chasing interaction"""
        y = self.ref_size // 2
        # Character being chased
        cv2.circle(img, (self.ref_size//4, y), 3, 1.0, 2)
        # Character chasing
        cv2.circle(img, (3*self.ref_size//4, y), 3, 1.0, 2)
        # Motion lines
        cv2.line(img, (self.ref_size//4+5, y), (3*self.ref_size//4-5, y), 0.6, 1)
    
    def _draw_following_interaction(self, img: np.ndarray):
        """Draw following interaction"""
        spacing = self.ref_size // 4
        y = self.ref_size // 2
        for i in range(3):
            x = spacing + i * spacing
            cv2.circle(img, (x, y), 2, 1.0, -1)
    
    def _draw_apart_interaction(self, img: np.ndarray):
        """Draw characters standing apart"""
        cv2.circle(img, (self.ref_size//6, self.ref_size//2), 3, 1.0, 2)
        cv2.circle(img, (5*self.ref_size//6, self.ref_size//2), 3, 1.0, 2)
    
    # Noise and distractor methods
    def _add_complex_background_noise(self, img: np.ndarray) -> np.ndarray:
        """Add complex background noise patterns"""
        noise_type = random.choice(["geometric", "texture", "lines", "dots"])
        
        if noise_type == "geometric":
            for _ in range(3):
                x, y = random.randint(5, self.ref_size-5), random.randint(5, self.ref_size-5)
                size = random.randint(2, 6)
                shape = random.choice(["circle", "square", "triangle"])
                if shape == "circle":
                    cv2.circle(img, (x, y), size, 0.3, 1)
                elif shape == "square":
                    cv2.rectangle(img, (x-size, y-size), (x+size, y+size), 0.3, 1)
        elif noise_type == "texture":
            noise = np.random.random((self.ref_size, self.ref_size)) * 0.2
            img = np.clip(img + noise, 0, 1)
        elif noise_type == "lines":
            for _ in range(5):
                x1, y1 = random.randint(0, self.ref_size), random.randint(0, self.ref_size)
                x2, y2 = random.randint(0, self.ref_size), random.randint(0, self.ref_size)
                cv2.line(img, (x1, y1), (x2, y2), 0.25, 1)
        elif noise_type == "dots":
            for _ in range(10):
                x, y = random.randint(0, self.ref_size), random.randint(0, self.ref_size)
                cv2.circle(img, (x, y), 1, 0.4, -1)
        
        return img
    
    def _add_character_distractors(self, img: np.ndarray, avoid_center: bool = True) -> np.ndarray:
        """Add distractor shapes that could be confused with character parts"""
        avoid_region = None
        if avoid_center:
            center = self.ref_size // 2
            avoid_region = (center-8, center-8, center+8, center+8)
        
        for _ in range(random.randint(1, 3)):
            attempts = 0
            while attempts < 10:
                x = random.randint(3, self.ref_size-3)
                y = random.randint(3, self.ref_size-3)
                
                if avoid_region is None or not (avoid_region[0] <= x <= avoid_region[2] and 
                                               avoid_region[1] <= y <= avoid_region[3]):
                    break
                attempts += 1
            
            distractor = random.choice(["small_circle", "small_square", "line"])
            size = random.randint(2, 5)
            
            if distractor == "small_circle":
                cv2.circle(img, (x, y), size, 0.4, 1)
            elif distractor == "small_square":
                cv2.rectangle(img, (x-size, y-size), (x+size, y+size), 0.4, 1)
            elif distractor == "line":
                angle = random.uniform(0, 2*np.pi)
                dx, dy = int(size * np.cos(angle)), int(size * np.sin(angle))
                cv2.line(img, (x-dx, y-dy), (x+dx, y+dy), 0.4, 1)
        
        return img
    
    # Additional noise methods for other reference types
    def _add_pose_background_noise(self, img: np.ndarray) -> np.ndarray:
        """Add pose-specific background noise"""
        # Add motion-like patterns
        for _ in range(3):
            x1, y1 = random.randint(0, self.ref_size), random.randint(0, self.ref_size)
            x2, y2 = x1 + random.randint(-10, 10), y1 + random.randint(-10, 10)
            cv2.line(img, (x1, y1), (x2, y2), 0.2, 1)
        return img
    
    def _add_pose_distractors(self, img: np.ndarray) -> np.ndarray:
        """Add pose-related distractors"""
        for _ in range(2):
            x, y = random.randint(5, self.ref_size-5), random.randint(5, self.ref_size-5)
            # Add random line that could be confused with limbs
            angle = random.uniform(0, 2*np.pi)
            length = random.randint(4, 8)
            dx, dy = int(length * np.cos(angle)), int(length * np.sin(angle))
            cv2.line(img, (x, y), (x+dx, y+dy), 0.5, 1)
        return img
    
    def _add_environmental_noise(self, img: np.ndarray) -> np.ndarray:
        """Add environmental background patterns"""
        # Add grid or natural patterns
        if random.random() < 0.5:
            # Grid pattern
            spacing = random.randint(6, 12)
            for i in range(0, self.ref_size, spacing):
                cv2.line(img, (i, 0), (i, self.ref_size), 0.15, 1)
                cv2.line(img, (0, i), (self.ref_size, i), 0.15, 1)
        else:
            # Organic patterns
            for _ in range(5):
                x, y = random.randint(0, self.ref_size), random.randint(0, self.ref_size)
                cv2.circle(img, (x, y), random.randint(1, 3), 0.2, -1)
        return img
    
    def _add_asset_distractors(self, img: np.ndarray, asset_type: str) -> np.ndarray:
        """Add distractors related to specific asset type"""
        # Add similar but different objects
        for _ in range(random.randint(1, 2)):
            x, y = random.randint(5, self.ref_size-5), random.randint(5, self.ref_size-5)
            
            if asset_type in ["tree", "house"]:
                # Add geometric shapes
                cv2.rectangle(img, (x-3, y-3), (x+3, y+3), 0.3, 1)
            elif asset_type in ["ball", "flower"]:
                # Add small circles
                cv2.circle(img, (x, y), 2, 0.3, 1)
            else:
                # Add random shape
                cv2.circle(img, (x, y), random.randint(2, 4), 0.3, 1)
        
        return img
    
    def _add_background_texture_noise(self, img: np.ndarray) -> np.ndarray:
        """Add texture noise for backgrounds"""
        noise = np.random.random((self.ref_size, self.ref_size)) * 0.15
        img = np.clip(img + noise, 0, 1)
        return img
    
    def _add_layout_background(self, img: np.ndarray) -> np.ndarray:
        """Add subtle background for layout references"""
        # Very light grid
        spacing = self.ref_size // 4
        for i in range(0, self.ref_size, spacing):
            cv2.line(img, (i, 0), (i, self.ref_size), 0.1, 1)
            cv2.line(img, (0, i), (self.ref_size, i), 0.1, 1)
        return img
    
    def _add_interaction_background(self, img: np.ndarray) -> np.ndarray:
        """Add background noise for interaction references"""
        # Add some random small elements
        for _ in range(3):
            x, y = random.randint(0, self.ref_size), random.randint(0, self.ref_size)
            cv2.circle(img, (x, y), 1, 0.2, -1)
        return img