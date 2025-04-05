import os
import random
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Sample a subset of images from each class folder.")
    parser.add_argument('--dataset-path', type=str, default='./NEU/train/images', help='Path to the dataset')
    parser.add_argument('--target-dir', type=str, default='./NEU_100', help='Directory to save sampled images')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of images to sample per class')
    return parser.parse_args()

def sample_images_per_class(dataset_path, target_dir, num_samples):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
    
    os.makedirs(target_dir, exist_ok=True)

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        if len(images) < num_samples:
            raise ValueError(f"Not enough images in class '{class_name}'. Found {len(images)}, needed {num_samples}.")
        
        selected_images = random.sample(images, num_samples)
        
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        
        for image_name in selected_images:
            src_path = os.path.join(class_dir, image_name)
            dst_path = os.path.join(target_class_dir, image_name)
            shutil.copyfile(src_path, dst_path)

    print(f"Selected {num_samples} samples from each class in '{dataset_path}' to '{target_dir}'.")

if __name__ == "__main__":
    args = parse_args()
    sample_images_per_class(args.dataset_path, args.target_dir, args.num_samples)