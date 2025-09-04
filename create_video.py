#!/usr/bin/env python3
"""
Simple script to convert images from inference_images directory into a video.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def create_video_from_images(image_dir="./inference_images", output_path="inference_video.mp4", fps=10):
    """
    Create a video from images in a directory.
    
    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Frames per second for the output video
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist")
        return
    
    # Get all image files and sort them by name
    image_files = sorted([f for f in image_dir.glob("*.png") if f.is_file()])
    
    if not image_files:
        print(f"No PNG images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Error: Could not read first image {image_files[0]}")
        return
    
    height, width, channels = first_image.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video at {fps} FPS...")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"Warning: Could not read image {image_file}")
            continue
        
        # Resize if necessary (in case images have different sizes)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Release everything
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved as: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert images to video")
    parser.add_argument("--image_dir", default="./inference_images", help="Directory containing images")
    parser.add_argument("--output", default="inference_video.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    
    args = parser.parse_args()
    
    create_video_from_images(args.image_dir, args.output, args.fps)


if __name__ == "__main__":
    main()