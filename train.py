import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from model.squeezenet_multi_scale_attention_zy_addition import SqueezeNetWithAttention as MSA_Addition_Pool35
from model.squeezenet_multi_scale_attention_yz_addition import SqueezeNetWithAttention as MSA_Addition_Pool53
from torchsummary import summary

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-scale attention network")
    parser.add_argument('--data-dir', type=str, default='./dataset', help='Base directory for datasets')
    parser.add_argument('--dataset', type=str, choices=['NEU', 'NEU_train_combine_noisy_images', 'NEU_100', 'NEU_50', 'NEU_10','malaria_dataset', 'malaria_dataset_2'], default='NEU', help='Dataset to use')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes in the dataset')
    parser.add_argument('--num-epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--log-file', type=str, default='./train_log.txt', help='File to log training progress')
    parser.add_argument('--model-type', type=str, choices=['MSA_Addition_Pool35', 'MSA_Addition_Pool53'], default='MSA_Addition_Pool35', help='Model type to use')
    parser.add_argument('--early-stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--checkpoint-path', type=str, default='./msa.pth', help='Path to save the best model checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset based on argument
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    dataset = ImageFolder(root=dataset_dir, transform=transform)
    print(f'Dataset classes: {dataset.classes}')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Choose model type
    if args.model_type == 'MSA_Addition_Pool35':
        model = MSA_Addition_Pool35(args.num_classes)
    elif args.model_type == 'MSA_Addition_Pool35':
        model = MSA_Addition_Pool53(args.num_classes)
    else:
        raise ValueError("Invalid model type")

    model = model.to(device)

    # Network architecture
    # print (model)
    # summary(model, input_size=(3, 224, 224)) # channel, h, w)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Remove existing log file
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    # Training and validation loop with early stopping
    best_val_accuracy = 0.0
    best_val_loss = float('inf')  # Initialize best val loss
    early_stop_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()

        val_accuracy = correct / total
        print(f'Epoch {epoch+1}/{args.num_epochs}, Training Loss: {total_loss / len(train_dataloader)}, Validation Loss: {val_loss / len(val_dataloader)}, Validation Accuracy: {val_accuracy}')

        with open(args.log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {total_loss / len(train_dataloader)}, Validation Loss: {val_loss / len(val_dataloader)}, Validation Accuracy: {val_accuracy}\n")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_dataloader),
                'val_accuracy': val_accuracy
            }, args.checkpoint_path)
            early_stop_counter = 0  # reset patience counter

        elif val_accuracy == best_val_accuracy:
            if (val_loss / len(val_dataloader)) < best_val_loss:
                best_val_loss = val_loss / len(val_dataloader)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / len(train_dataloader)
                }, args.checkpoint_path)
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop:
                print(f'Early stopping at epoch {epoch+1} as validation accuracy did not improve for {args.early_stop} epochs.')
                break
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop:
                print(f'Early stopping at epoch {epoch+1} as validation accuracy did not improve for {args.early_stop} epochs.')
                break

    print(f'Best Validation Accuracy: {best_val_accuracy}')

if __name__ == "__main__":
    main()
