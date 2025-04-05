import os
import random
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Split malaria dataset into train/test sets with random sampling.")
    parser.add_argument('--source-dir', type=str, default='./cell_images', help='Root directory containing Parasitized and Uninfected folders')
    parser.add_argument('--output-dir', type=str, default='./malaria_dataset', help='Output dataset directory')
    parser.add_argument('--train-count', type=int, default=200, help='Number of training images per class')
    parser.add_argument('--test-count', type=int, default=50, help='Number of testing images per class')
    return parser.parse_args()


def generate_random_dataset(parasite_images, uninfected_images, train_count, test_count):
    # Shuffle and sample training images
    train_parasite = random.sample(parasite_images, train_count)
    train_uninfected = random.sample(uninfected_images, train_count)

    # Remaining images for test set
    remaining_parasite = list(set(parasite_images) - set(train_parasite))
    remaining_uninfected = list(set(uninfected_images) - set(train_uninfected))

    test_parasite = random.sample(remaining_parasite, test_count)
    test_uninfected = random.sample(remaining_uninfected, test_count)

    print(f"Training set - Parasite: {len(train_parasite)}, Uninfected: {len(train_uninfected)}")
    print(f"Testing set - Parasite: {len(test_parasite)}, Uninfected: {len(test_uninfected)}")

    return train_parasite, train_uninfected, test_parasite, test_uninfected


def save_to_dataset(image_list, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for image_name in image_list:
        src = os.path.join(source_dir, image_name)
        dst = os.path.join(target_dir, image_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def main():
    args = parse_args()

    parasite_dir = os.path.join(args.source_dir, 'Parasitized')
    uninfected_dir = os.path.join(args.source_dir, 'Uninfected')

    parasite_images = [f for f in os.listdir(parasite_dir) if f.endswith('.png')]
    uninfected_images = [f for f in os.listdir(uninfected_dir) if f.endswith('.png')]

    if len(parasite_images) < args.train_count + args.test_count:
        raise ValueError("Not enough parasite images to fulfill request.")
    if len(uninfected_images) < args.train_count + args.test_count:
        raise ValueError("Not enough uninfected images to fulfill request.")

    train_p, train_u, test_p, test_u = generate_random_dataset(
        parasite_images, uninfected_images, args.train_count, args.test_count
    )

    save_to_dataset(train_p, parasite_dir, os.path.join(args.output_dir, 'train', 'Parasitized'))
    save_to_dataset(train_u, uninfected_dir, os.path.join(args.output_dir, 'train', 'Uninfected'))
    save_to_dataset(test_p, parasite_dir, os.path.join(args.output_dir, 'test', 'Parasitized'))
    save_to_dataset(test_u, uninfected_dir, os.path.join(args.output_dir, 'test', 'Uninfected'))

    print(f"Dataset created at '{args.output_dir}'")


if __name__ == "__main__":
    main()
