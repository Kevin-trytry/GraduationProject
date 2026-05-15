"""
YOLO Hand Detection Inference Script
====================================
Use your trained YOLO model to detect hands in images and videos.
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model path - Update this to your trained model
MODEL_PATH = "./output/hand_detection_model/weights/best.pt"

# Inference settings
CONFIDENCE_THRESHOLD = 0.25        # Confidence threshold for detections
IOU_THRESHOLD = 0.45               # IoU threshold for NMS (Non-Maximum Suppression)
DEVICE = 0                         # GPU device (0 for first GPU)

# Input/Output paths
INPUT_IMAGE_PATH = None            # Set to image path for single image
INPUT_VIDEO_PATH = None            # Set to video path for video inference
INPUT_DIRECTORY = None             # Set to directory path for batch processing
OUTPUT_DIR = "./output/predictions"

# Class names (must match your training classes)
CLASSES = {0: 'Hand_0', 1: 'Hand_1', 2: 'Hand_2', 3: 'Hand_3', 4: 'Hand_4',
           5: 'Hand_5', 6: 'Hand_6', 7: 'Hand_7', 8: 'Hand_8', 9: 'Hand_9', 10: 'Hand_B'}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(model_path, device):
    """
    Load trained YOLO model.
    
    Args:
        model_path (str): Path to trained model weights
        device (int): GPU device ID
        
    Returns:
        YOLO: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"✓ Model loaded successfully on device {device}")
    return model


def predict_on_image(model, image_path, conf_threshold, iou_threshold, device):
    """
    Run inference on a single image.
    
    Args:
        model: YOLO model
        image_path (str): Path to image file
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
        device (int): GPU device
        
    Returns:
        list: Detection results
    """
    print(f"\n🖼️  Processing image: {image_path}")
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False
    )
    
    return results


def predict_on_video(model, video_path, conf_threshold, iou_threshold, device, 
                    output_path=None):
    """
    Run inference on video and optionally save output video.
    
    Args:
        model: YOLO model
        video_path (str): Path to video file
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
        device (int): GPU device
        output_path (str): Path to save output video
    """
    print(f"\n🎥 Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"📹 Saving output to: {output_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        if writer:
            writer.write(annotated_frame)
        
        # Display frame
        cv2.imshow('Hand Detection - Press Q to exit', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Video processing complete. Total frames: {frame_count}")


def predict_on_directory(model, directory_path, conf_threshold, iou_threshold, 
                        device, output_dir):
    """
    Run inference on all images in a directory.
    
    Args:
        model: YOLO model
        directory_path (str): Path to directory
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
        device (int): GPU device
        output_dir (str): Directory to save results
    """
    print(f"\n📁 Batch processing directory: {directory_path}")
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    # Collect all image files
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_formats:
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_path}")
        
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )
        
        # Save annotated image
        result = results[0]
        if result.boxes:  # If detections found
            annotated_frame = result.plot()
            output_filename = f"{Path(image_path).stem}_detected.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, annotated_frame)
            print(f"   ✓ Saved: {output_path}")
            print(f"   Detections: {len(result.boxes)} hand(s)")
        else:
            print(f"   No hands detected")
    
    print(f"\n✓ Batch processing complete. Results saved to: {output_dir}")


def print_detection_stats(results):
    """
    Print statistics about detections.
    
    Args:
        results (list): List of detection results
    """
    print("\n📊 Detection Statistics:")
    print("-" * 50)
    
    total_detections = 0
    class_counts = defaultdict(int)
    confidences = []
    
    for result in results:
        for box in result.boxes:
            total_detections += 1
            class_id = int(box.cls)
            class_counts[class_id] += 1
            confidences.append(float(box.conf))
    
    if total_detections > 0:
        print(f"Total detections: {total_detections}")
        
        # Class distribution
        print("\nClass distribution:")
        for class_id in sorted(class_counts.keys()):
            class_name = CLASSES.get(class_id, f"Class {class_id}")
            count = class_counts[class_id]
            percentage = (count / total_detections) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Confidence statistics
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        print(f"\nConfidence statistics:")
        print(f"  Average: {avg_conf:.3f}")
        print(f"  Min: {min_conf:.3f}")
        print(f"  Max: {max_conf:.3f}")
    else:
        print("No detections found")
    
    print("-" * 50)


def main():
    """
    Main inference execution.
    """
    print("=" * 70)
    print("🎯 YOLO Hand Detection Inference")
    print("=" * 70)
    
    # Load model
    model = load_model(MODEL_PATH, DEVICE)
    
    # Set confidence and IoU thresholds
    model.conf = CONFIDENCE_THRESHOLD
    model.iou = IOU_THRESHOLD
    
    print(f"\n⚙️  Inference Settings:")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   IoU threshold: {IOU_THRESHOLD}")
    
    # Choose inference mode
    if INPUT_IMAGE_PATH and os.path.exists(INPUT_IMAGE_PATH):
        results = predict_on_image(model, INPUT_IMAGE_PATH, CONFIDENCE_THRESHOLD,
                                  IOU_THRESHOLD, DEVICE)
        print_detection_stats(results)
        
        # Display result
        result = results[0]
        annotated_frame = result.plot()
        cv2.imshow('Hand Detection Result', annotated_frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif INPUT_VIDEO_PATH and os.path.exists(INPUT_VIDEO_PATH):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_video = os.path.join(OUTPUT_DIR, 'output_video.mp4')
        predict_on_video(model, INPUT_VIDEO_PATH, CONFIDENCE_THRESHOLD,
                        IOU_THRESHOLD, DEVICE, output_video)
    
    elif INPUT_DIRECTORY and os.path.exists(INPUT_DIRECTORY):
        predict_on_directory(model, INPUT_DIRECTORY, CONFIDENCE_THRESHOLD,
                            IOU_THRESHOLD, DEVICE, OUTPUT_DIR)
    
    else:
        print("\n⚠️  No input specified!")
        print("Please set one of the following in the script:")
        print("  - INPUT_IMAGE_PATH: for single image inference")
        print("  - INPUT_VIDEO_PATH: for video inference")
        print("  - INPUT_DIRECTORY: for batch image processing")
        print("\nExample usage:")
        print("  INPUT_IMAGE_PATH = './image.jpg'")
        print("  OUTPUT_DIR = './results'")
        print("  Then run: python inference.py")
    
    print("\n✅ Inference complete!")
    print("=" * 70)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_single_image():
    """Run inference on single image"""
    model = YOLO(MODEL_PATH)
    results = model.predict(source='./image.jpg', conf=0.25)
    
    # Visualize
    result = results[0]
    annotated_frame = result.plot()
    cv2.imshow('Detections', annotated_frame)
    cv2.waitKey(0)


def example_batch_prediction():
    """Run inference on directory and collect all detections"""
    model = YOLO(MODEL_PATH)
    results = model.predict(source='./data handling/New data set', conf=0.25)
    
    for result in results:
        print(f"Image: {result.path}")
        for box in result.boxes:
            print(f"  Class: {box.cls}, Confidence: {box.conf}")
            print(f"  Box: {box.xyxy}")


def example_with_custom_output():
    """Run inference and save annotated images"""
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source='./data handling/New data set',
        conf=0.25,
        save=True,           # Save annotated images
        save_dir='./predictions'
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
    
    # Uncomment to run examples:
    # example_single_image()
    # example_batch_prediction()
    # example_with_custom_output()
