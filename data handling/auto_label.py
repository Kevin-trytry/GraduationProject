"""
Auto-Labeling Script for Hand Detection in YOLO Format
========================================================
This script automatically detects hands in raw images using MediaPipe Hand Landmarker
and generates YOLO format bounding box labels (.txt files).

Features:
- Detects hands using MediaPipe Hand Landmarker
- Generates YOLO format normalized bounding boxes
- Adds 15-20% padding around detected hands
- Creates empty .txt files for background images (no hands detected)
- Processes all images in subdirectories
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# Resolve base dir so the script works regardless of which directory it is run from
_BASE = os.path.dirname(os.path.abspath(__file__))  # = .../data handling/


# ============================================================================
# CONFIGURATION - Modify these paths and parameters as needed
# ============================================================================
RAW_DATASET_PATH      = os.path.join(_BASE, "New data set")       # Subfolders: 0/, 1/, ..., B/
OUTPUT_LABELS_PATH    = os.path.join(_BASE, "labels")              # Output .txt label files
HAND_LANDMARKER_MODEL = os.path.join(_BASE, "hand_landmarker.task")  # MediaPipe model

# Hand detection parameters
CONFIDENCE_THRESHOLD = 0.3      # Confidence threshold (lowered to 0.3 to catch more hand poses)
PADDING_PERCENT = 0.175         # Padding percentage: 0.175 = 17.5% (between 15-20%)

# Set to True to only reprocess images whose label file is currently empty
RERUN_EMPTY_ONLY = True

# Class mapping: folder name -> class ID (must match CLASSES in train_yolo.py)
CLASS_MAP = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'B': 10
}

# Image processing
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_hand_detector(model_path):
    """
    Load MediaPipe Hand Landmarker model.
    
    Args:
        model_path (str): Path to the hand_landmarker.task file
        
    Returns:
        HandLandmarker: Loaded hand detection model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


def detect_hands(image, detector, confidence_threshold=0.5):
    """
    Detect hands in an image using MediaPipe Hand Landmarker.
    
    Args:
        image (np.ndarray): Input image (BGR format)
        detector: MediaPipe HandLandmarker object
        confidence_threshold (float): Minimum confidence for detection
        
    Returns:
        list: List of hand detections with confidence scores
    """
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect hands
    detection_result = detector.detect(mp_image)
    
    # Filter by confidence threshold
    hand_detections = []
    if detection_result.handedness:
        for i, handedness_list in enumerate(detection_result.handedness):
            if handedness_list[0].score >= confidence_threshold:
                hand_detections.append(i)
    
    return hand_detections, detection_result


def get_bounding_box_from_landmarks(landmarks, image_height, image_width, padding_percent=0.175):
    """
    Calculate bounding box from hand landmarks with padding.
    
    Args:
        landmarks: Hand landmarks from MediaPipe
        image_height (int): Image height in pixels
        image_width (int): Image width in pixels
        padding_percent (float): Padding percentage to apply
        
    Returns:
        tuple: (x_center, y_center, width, height) in normalized YOLO format
    """
    # Extract x, y coordinates from landmarks
    # Note: MediaPipe Tasks API returns landmarks as a plain list directly
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    # Get bounding box extremes (normalized)
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Calculate width and height
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    # Add padding (15-20%)
    padding_x = bbox_width * padding_percent
    padding_y = bbox_height * padding_percent
    
    x_min = max(0, x_min - padding_x)
    x_max = min(1, x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(1, y_max + padding_y)
    
    # Convert to YOLO format (center x, center y, width, height in normalized format)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def save_yolo_labels(image_path, detector, output_dir, confidence_threshold, padding_percent, class_id):
    """
    Process a single image and save YOLO format labels.
    
    Args:
        image_path (str): Path to the image file
        detector: MediaPipe HandLandmarker object
        output_dir (str): Directory to save label files
        confidence_threshold (float): Confidence threshold for detection
        padding_percent (float): Padding percentage for bounding box
        class_id (int): YOLO class ID derived from folder name
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to read image: {image_path}")
            return False
        
        image_height, image_width = image.shape[:2]
        
        # Detect hands
        hand_detections, detection_result = detect_hands(image, detector, confidence_threshold)
        
        # Generate output label file path
        image_filename = Path(image_path).stem
        label_filename = f"{image_filename}.txt"
        label_path = os.path.join(output_dir, label_filename)
        
        # Save labels
        if hand_detections:
            # Hand(s) detected - save bounding boxes with correct class ID
            with open(label_path, 'w') as f:
                for hand_idx in hand_detections:
                    landmarks = detection_result.hand_landmarks[hand_idx]
                    x_center, y_center, width, height = get_bounding_box_from_landmarks(
                        landmarks, image_height, image_width, padding_percent
                    )
                    # YOLO format: class_id x_center y_center width height (all normalized 0-1)
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            print(f"✓ {label_filename} - class={class_id}, {len(hand_detections)} hand(s) detected")
        else:
            # No hands detected - create empty file (background/negative sample)
            with open(label_path, 'w') as f:
                pass  # Create empty file
            print(f"◯ {label_filename} - Background (no hands detected)")
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False


def process_dataset(raw_dataset_path, output_labels_path, model_path, 
                   confidence_threshold=0.5, padding_percent=0.175):
    """
    Process entire dataset and generate YOLO format labels.
    
    Args:
        raw_dataset_path (str): Path to raw image dataset
        output_labels_path (str): Path to save label files
        model_path (str): Path to hand landmarker model
        confidence_threshold (float): Confidence threshold for detection
        padding_percent (float): Padding percentage for bounding box
    """
    print("=" * 70)
    print("Starting Hand Detection and YOLO Label Generation")
    print("=" * 70)
    
    # Validate paths
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {raw_dataset_path}")
    
    # Create output directory
    os.makedirs(output_labels_path, exist_ok=True)
    
    # Load detector
    print(f"\n📦 Loading MediaPipe Hand Landmarker model...")
    detector = load_hand_detector(model_path)
    print(f"✓ Model loaded successfully")
    
    # Process all images
    total_images = 0
    processed_images = 0
    
    print(f"\n📁 Scanning dataset: {raw_dataset_path}")
    print("-" * 70)
    
    # Recursively process all subdirectories
    for root, dirs, files in os.walk(raw_dataset_path):
        # Determine class_id from the immediate subfolder name (e.g. '0', '1', ..., 'B')
        folder_name = os.path.basename(root)
        class_id = CLASS_MAP.get(folder_name, None)
        
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_FORMATS:
                total_images += 1
                image_path = os.path.join(root, filename)
                
                if class_id is None:
                    # Image is in root or unknown folder - skip
                    print(f"⚠️  Skipping {filename} (unknown folder '{folder_name}', not in CLASS_MAP)")
                    continue
                
                # If RERUN_EMPTY_ONLY, skip images that already have a non-empty label
                if RERUN_EMPTY_ONLY:
                    label_path = os.path.join(output_labels_path, Path(image_path).stem + '.txt')
                    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                        continue  # Already labeled, skip
                
                success = save_yolo_labels(
                    image_path, detector, output_labels_path,
                    confidence_threshold, padding_percent, class_id
                )
                if success:
                    processed_images += 1
    
    # Summary
    print("-" * 70)
    print(f"\nProcessing Complete!")
    print(f"   Total images: {total_images}")
    print(f"   Successfully processed: {processed_images}")
    print(f"   Labels saved to: {output_labels_path}")
    print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    process_dataset(
        raw_dataset_path=RAW_DATASET_PATH,
        output_labels_path=OUTPUT_LABELS_PATH,
        model_path=HAND_LANDMARKER_MODEL,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        padding_percent=PADDING_PERCENT
    )
