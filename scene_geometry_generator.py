import numpy as np
import cv2
import random
from typing import Dict, List, Tuple, Optional
import pickle
import os
import json

class SceneGeometryGenerator:
    def __init__(self, canvas_size=256, ref_size=64):
        self.canvas_size = canvas_size
        self.ref_size = ref_size
        
        # Character definitions
        self.character_types = {
            "dog": {"head": "circle", "body": "oval", "size_mult": 1.0},
            "cat": {"head": "circle", "body": "oval", "size_mult": 0.8},
            "bird": {"head": "circle", "body": "triangle", "size_mult": 0.6},
            "robot": {"head": "square", "body": "rectangle", "size_mult": 1.2},
            "alien": {"head": "oval", "body": "triangle", "size_mult": 0.9}
        }
        
        # Asset types (objects in scene)
        self.asset_types = {
            "tree": {"shape": "triangle_on_rectangle", "size": "large"},
            "house": {"shape": "rectangle_with_triangle_roof", "size": "large"},
            "ball": {"shape": "circle", "size": "small"},
            "car": {"shape": "rectangle_with_circles", "size": "medium"},
            "flower": {"shape": "small_star", "size": "small"},
            "rock": {"shape": "irregular_oval", "size": "small"},
            "cloud": {"shape": "fluffy_oval", "size": "medium"},
            "sun": {"shape": "circle_with_rays", "size": "medium"}
        }
        
        # Background types
        self.background_types = {
            "park": {"ground": "grass", "sky": "blue", "features": ["trees", "flowers"]},
            "city": {"ground": "street", "sky": "blue", "features": ["houses", "cars"]},
            "beach": {"ground": "sand", "sky": "blue", "features": ["rocks", "clouds"]},
            "space": {"ground": "flat", "sky": "dark", "features": ["stars", "planets"]},
            "forest": {"ground": "grass", "sky": "green", "features": ["trees", "rocks"]}
        }
        
        # Character poses and interactions
        self.poses = ["standing", "sitting", "walking", "running", "jumping"]
        self.interactions = ["facing_each_other", "playing_together", "chasing", "following", "standing_apart"]
        
        # Spatial relationships
        self.spatial_relations = ["left_of", "right_of", "behind", "in_front_of", "next_to", "far_from"]
        
    def generate_scene_layout(self, num_characters: int, num_assets: int, background_type: str) -> Dict:
        """Generate spatial layout for entire scene"""
        
        # Divide canvas into regions for better spatial planning
        regions = self._create_spatial_regions()
        
        # Generate characters with positions and relationships
        characters = []
        for i in range(num_characters):
            char_type = random.choice(list(self.character_types.keys()))
            pose = random.choice(self.poses)
            
            # Assign character to region (foreground/midground)
            region = random.choice(["foreground", "midground"])
            position = self._get_position_in_region(region, regions)
            
            character = {
                "id": f"char_{i}",
                "type": char_type,
                "pose": pose,
                "position": position,
                "region": region,
                "size": self.character_types[char_type]["size_mult"]
            }
            characters.append(character)
        
        # Generate spatial relationships between characters
        relationships = self._generate_character_relationships(characters)
        
        # Generate assets (objects) with positions
        assets = []
        for i in range(num_assets):
            asset_type = random.choice(list(self.asset_types.keys()))
            
            # Choose region based on asset type
            if asset_type in ["tree", "house"]:
                region = "background"
            elif asset_type in ["ball", "flower"]:
                region = "foreground"
            else:
                region = random.choice(["midground", "background"])
            
            position = self._get_position_in_region(region, regions)
            
            asset = {
                "id": f"asset_{i}",
                "type": asset_type,
                "position": position,
                "region": region,
                "size": self.asset_types[asset_type]["size"]
            }
            assets.append(asset)
        
        # Generate background
        background = {
            "type": background_type,
            "features": self.background_types[background_type]["features"]
        }
        
        scene_layout = {
            "characters": characters,
            "assets": assets,
            "background": background,
            "relationships": relationships,
            "canvas_size": self.canvas_size
        }
        
        return scene_layout
    
    def _create_spatial_regions(self) -> Dict:
        """Define spatial regions in the scene"""
        h, w = self.canvas_size, self.canvas_size
        
        regions = {
            "foreground": {"y_range": (h*0.6, h*0.9), "x_range": (w*0.1, w*0.9)},
            "midground": {"y_range": (h*0.3, h*0.6), "x_range": (w*0.2, w*0.8)},
            "background": {"y_range": (h*0.1, h*0.4), "x_range": (w*0.1, w*0.9)},
            "sky": {"y_range": (0, h*0.3), "x_range": (0, w)}
        }
        
        return regions
    
    def _get_position_in_region(self, region: str, regions: Dict) -> Tuple[int, int]:
        """Get random position within specified region"""
        y_min, y_max = regions[region]["y_range"]
        x_min, x_max = regions[region]["x_range"]
        
        x = random.randint(int(x_min), int(x_max))
        y = random.randint(int(y_min), int(y_max))
        
        return (x, y)
    
    def _generate_character_relationships(self, characters: List[Dict]) -> List[Dict]:
        """Generate spatial and interaction relationships between characters"""
        relationships = []
        
        for i in range(len(characters)):
            for j in range(i + 1, len(characters)):
                char1, char2 = characters[i], characters[j]
                
                # Determine spatial relationship
                spatial_rel = self._determine_spatial_relationship(char1["position"], char2["position"])
                
                # Determine interaction type
                interaction = random.choice(self.interactions)
                
                relationship = {
                    "char1_id": char1["id"],
                    "char2_id": char2["id"],
                    "spatial": spatial_rel,
                    "interaction": interaction
                }
                relationships.append(relationship)
        
        return relationships
    
    def _determine_spatial_relationship(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> str:
        """Determine spatial relationship between two positions"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        dx, dy = x2 - x1, y2 - y1
        
        if abs(dx) > abs(dy):
            return "right_of" if dx > 0 else "left_of"
        else:
            return "behind" if dy < 0 else "in_front_of"
    
    def draw_character(self, canvas: np.ndarray, character: Dict, color: float = 1.0):
        """Draw a character on the canvas"""
        char_type = character["type"]
        pos_x, pos_y = character["position"]
        size_mult = character["size"]
        pose = character["pose"]
        
        char_def = self.character_types[char_type]
        base_size = int(15 * size_mult)
        
        # Adjust position based on region (perspective)
        if character["region"] == "background":
            base_size = int(base_size * 0.6)
        elif character["region"] == "foreground":
            base_size = int(base_size * 1.2)
        
        # Draw head
        head_y = pos_y - base_size
        if char_def["head"] == "circle":
            cv2.circle(canvas, (pos_x, head_y), base_size//2, color, 2)
        elif char_def["head"] == "square":
            hs = base_size//2
            cv2.rectangle(canvas, (pos_x-hs, head_y-hs), (pos_x+hs, head_y+hs), color, 2)
        elif char_def["head"] == "oval":
            cv2.ellipse(canvas, (pos_x, head_y), (base_size//2, int(base_size*0.35)), 0, 0, 360, color, 2)
        
        # Draw body
        body_y = pos_y
        if char_def["body"] == "oval":
            cv2.ellipse(canvas, (pos_x, body_y), (base_size//3, base_size//2), 0, 0, 360, color, 2)
        elif char_def["body"] == "rectangle":
            bs = base_size//3
            cv2.rectangle(canvas, (pos_x-bs, body_y-base_size//2), (pos_x+bs, body_y+base_size//2), color, 2)
        elif char_def["body"] == "triangle":
            pts = np.array([[pos_x, body_y-base_size//2], 
                           [pos_x-base_size//3, body_y+base_size//2],
                           [pos_x+base_size//3, body_y+base_size//2]], np.int32)
            cv2.polylines(canvas, [pts], True, color, 2)
        
        # Draw pose-specific elements (arms, legs)
        self._draw_pose_elements(canvas, pos_x, pos_y, base_size, pose, color)
    
    def _draw_pose_elements(self, canvas: np.ndarray, pos_x: int, pos_y: int, 
                           base_size: int, pose: str, color: float):
        """Draw pose-specific limbs and elements"""
        arm_length = base_size // 2
        leg_length = base_size // 2
        
        # Arms
        arm_y = pos_y - base_size // 4
        if pose == "standing":
            cv2.line(canvas, (pos_x, arm_y), (pos_x - arm_length//2, arm_y + arm_length), color, 2)
            cv2.line(canvas, (pos_x, arm_y), (pos_x + arm_length//2, arm_y + arm_length), color, 2)
        elif pose == "walking" or pose == "running":
            cv2.line(canvas, (pos_x, arm_y), (pos_x - arm_length, arm_y), color, 2)
            cv2.line(canvas, (pos_x, arm_y), (pos_x + arm_length//2, arm_y - arm_length//2), color, 2)
        elif pose == "jumping":
            cv2.line(canvas, (pos_x, arm_y), (pos_x - arm_length, arm_y - arm_length//2), color, 2)
            cv2.line(canvas, (pos_x, arm_y), (pos_x + arm_length, arm_y - arm_length//2), color, 2)
        
        # Legs
        leg_y = pos_y + base_size // 2
        if pose == "standing":
            cv2.line(canvas, (pos_x, leg_y), (pos_x - leg_length//3, leg_y + leg_length), color, 2)
            cv2.line(canvas, (pos_x, leg_y), (pos_x + leg_length//3, leg_y + leg_length), color, 2)
        elif pose == "sitting":
            cv2.line(canvas, (pos_x, leg_y), (pos_x - leg_length, leg_y), color, 2)
            cv2.line(canvas, (pos_x, leg_y), (pos_x + leg_length, leg_y), color, 2)
        elif pose == "walking":
            cv2.line(canvas, (pos_x, leg_y), (pos_x - leg_length//2, leg_y + leg_length), color, 2)
            cv2.line(canvas, (pos_x, leg_y), (pos_x + leg_length//3, leg_y + leg_length//2), color, 2)
        elif pose == "running":
            cv2.line(canvas, (pos_x, leg_y), (pos_x - leg_length, leg_y + leg_length//2), color, 2)
            cv2.line(canvas, (pos_x, leg_y), (pos_x + leg_length//2, leg_y + leg_length), color, 2)
        elif pose == "jumping":
            cv2.line(canvas, (pos_x, leg_y), (pos_x - leg_length//2, leg_y - leg_length//3), color, 2)
            cv2.line(canvas, (pos_x, leg_y), (pos_x + leg_length//2, leg_y - leg_length//3), color, 2)
    
    def draw_asset(self, canvas: np.ndarray, asset: Dict, color: float = 0.7):
        """Draw an asset/object on the canvas"""
        asset_type = asset["type"]
        pos_x, pos_y = asset["position"]
        size = asset["size"]
        
        # Size mapping
        size_map = {"small": 8, "medium": 15, "large": 25}
        base_size = size_map[size]
        
        # Adjust size based on region (perspective)
        if asset["region"] == "background":
            base_size = int(base_size * 0.7)
        elif asset["region"] == "foreground":
            base_size = int(base_size * 1.1)
        
        asset_def = self.asset_types[asset_type]
        shape = asset_def["shape"]
        
        if shape == "circle":
            cv2.circle(canvas, (pos_x, pos_y), base_size, color, 2)
        elif shape == "triangle_on_rectangle":  # Tree
            # Trunk
            trunk_w, trunk_h = base_size//4, base_size//2
            cv2.rectangle(canvas, (pos_x-trunk_w, pos_y), (pos_x+trunk_w, pos_y+trunk_h), color, 2)
            # Leaves
            leaf_pts = np.array([[pos_x, pos_y-base_size], 
                               [pos_x-base_size//2, pos_y],
                               [pos_x+base_size//2, pos_y]], np.int32)
            cv2.polylines(canvas, [leaf_pts], True, color, 2)
        elif shape == "rectangle_with_triangle_roof":  # House
            # Base
            house_w, house_h = base_size, base_size//2
            cv2.rectangle(canvas, (pos_x-house_w//2, pos_y-house_h), (pos_x+house_w//2, pos_y), color, 2)
            # Roof
            roof_pts = np.array([[pos_x, pos_y-house_h-base_size//3], 
                               [pos_x-house_w//2, pos_y-house_h],
                               [pos_x+house_w//2, pos_y-house_h]], np.int32)
            cv2.polylines(canvas, [roof_pts], True, color, 2)
        elif shape == "rectangle_with_circles":  # Car
            # Body
            car_w, car_h = base_size, base_size//3
            cv2.rectangle(canvas, (pos_x-car_w//2, pos_y-car_h), (pos_x+car_w//2, pos_y), color, 2)
            # Wheels
            wheel_r = base_size//6
            cv2.circle(canvas, (pos_x-car_w//3, pos_y), wheel_r, color, 2)
            cv2.circle(canvas, (pos_x+car_w//3, pos_y), wheel_r, color, 2)
        elif shape == "small_star":  # Flower
            star_pts = []
            for i in range(8):
                angle = i * np.pi / 4
                r = base_size if i % 2 == 0 else base_size // 2
                x = pos_x + int(r * np.cos(angle))
                y = pos_y + int(r * np.sin(angle))
                star_pts.append([x, y])
            star_pts = np.array(star_pts, np.int32)
            cv2.polylines(canvas, [star_pts], True, color, 2)
        elif shape == "circle_with_rays":  # Sun
            cv2.circle(canvas, (pos_x, pos_y), base_size//2, color, 2)
            # Rays
            for i in range(8):
                angle = i * np.pi / 4
                x1 = pos_x + int((base_size//2 + 3) * np.cos(angle))
                y1 = pos_y + int((base_size//2 + 3) * np.sin(angle))
                x2 = pos_x + int((base_size//2 + 8) * np.cos(angle))
                y2 = pos_y + int((base_size//2 + 8) * np.sin(angle))
                cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
        else:  # Default to oval
            cv2.ellipse(canvas, (pos_x, pos_y), (base_size, base_size//2), 0, 0, 360, color, 2)
    
    def draw_background(self, canvas: np.ndarray, background: Dict):
        """Draw background elements"""
        bg_type = background["type"]
        bg_def = self.background_types[bg_type]
        
        h, w = self.canvas_size, self.canvas_size
        
        # Draw ground line
        ground_y = int(h * 0.7)
        cv2.line(canvas, (0, ground_y), (w, ground_y), 0.5, 2)
        
        # Add background-specific elements
        if bg_type == "park":
            # Add some grass texture
            for i in range(10):
                x = random.randint(0, w)
                y = random.randint(ground_y, h)
                cv2.line(canvas, (x, y), (x, y-5), 0.3, 1)
        elif bg_type == "city":
            # Add building silhouettes
            for i in range(3):
                x = random.randint(0, w)
                building_h = random.randint(30, 60)
                cv2.rectangle(canvas, (x, ground_y-building_h), (x+20, ground_y), 0.4, 2)
        elif bg_type == "space":
            # Add stars
            for i in range(20):
                x, y = random.randint(0, w), random.randint(0, int(h*0.5))
                cv2.circle(canvas, (x, y), 1, 0.6, -1)
    
    def create_scene_sketch(self, scene_layout: Dict) -> np.ndarray:
        """Create the complete scene sketch from layout"""
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        
        # Draw background first
        self.draw_background(canvas, scene_layout["background"])
        
        # Draw assets (background to foreground)
        assets_by_region = {"background": [], "midground": [], "foreground": []}
        for asset in scene_layout["assets"]:
            assets_by_region[asset["region"]].append(asset)
        
        for region in ["background", "midground", "foreground"]:
            for asset in assets_by_region[region]:
                self.draw_asset(canvas, asset)
        
        # Draw characters (background to foreground)
        chars_by_region = {"background": [], "midground": [], "foreground": []}
        for character in scene_layout["characters"]:
            chars_by_region[character["region"]].append(character)
        
        for region in ["background", "midground", "foreground"]:
            for character in chars_by_region[region]:
                self.draw_character(canvas, character)
        
        return canvas