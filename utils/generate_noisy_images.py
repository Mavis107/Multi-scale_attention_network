import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Generate noisy images for data augmentation")
    parser.add_argument('--data-dir', type=str, default='./NEU/train/images', help='Directory containing input images')
    parser.add_argument('--save-dir', type=str, default='./NEU_train_combine_noisy_images', help='Directory to save noisy images')
    parser.add_argument('--snr-db', type=float, default=20.0, help='Signal-to-noise ratio in decibels')
    parser.add_argument('--num-noisy-images', type=int, default=5, help='Number of noisy images to generate per input')
    parser.add_argument('--image-size', type=int, default=224, help='Image size for resizing (square)')
    return parser.parse_args()

def generate_noise_image(signal_image, snr_db=20, num_images=5):
    signal_power = np.mean(signal_image ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noisy_images = []
    for _ in range(num_images):
        noise = np.random.normal(scale=np.sqrt(noise_power), size=signal_image.shape)
        noisy_image = signal_image + noise
        noisy_images.append(noisy_image)
    return np.stack(noisy_images)

def main():
    args = parse_args()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = ImageFolder(root=args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.save_dir, exist_ok=True)

    # Generate and save images
    for idx, (images, labels_test) in enumerate(loader):
        image_np = images.numpy()[0]
        label = labels_test.item()

        # Generate noisy images
        noisy_imgs = generate_noise_image(image_np, snr_db=args.snr_db, num_images=args.num_noisy_images)

        label_dir = os.path.join(args.save_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # Save noisy images
        for j in range(args.num_noisy_images):
            img = noisy_imgs[j]
            img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
            save_path = os.path.join(label_dir, f"noisy_image_{idx * args.num_noisy_images + j}.png")
            plt.imsave(save_path, img_norm.transpose(1, 2, 0))

        # Save original image
        orig_img = image_np.transpose(1, 2, 0)
        plt.imsave(os.path.join(label_dir, f"original_image_{idx}.png"), orig_img)

if __name__ == "__main__":
    main()