import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model.squeezenet_multi_scale_attention_zy_addition import SqueezeNetWithAttention as MSA_Addition_Pool35
from model.squeezenet_multi_scale_attention_yz_addition import SqueezeNetWithAttention as MSA_Addition_Pool53

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./msa.pth', help='Path to the saved model file (.pth)')
    parser.add_argument('--model_type', type=str, choices=['MSA_Addition_Pool35', 'MSA_Addition_Pool53'], default='MSA_Addition_Pool35', help='Type of the model to use')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes in the dataset')
    parser.add_argument('--data-dir', type=str, default='./inference', help='Base directory for datasets')
    parser.add_argument('--dataset', type=str, choices=['NEU_test', 'NEU_test_with_noisy_images', 'malaria_test', 'malaria_test_2'], default='NEU_test', help='Dataset to use')
    return parser.parse_args()

def load_model(args):
    if args.model_type == 'MSA_Addition_Pool35':
        model = MSA_Addition_Pool35(args.num_classes)
    elif args.model_type == 'MSA_Addition_Pool53':
        model = MSA_Addition_Pool53(args.num_classes)


    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']}")
    model = model.to(device)
    return model

def evaluate_model(model, test_dataloader):
    criterion_test = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    model.eval()
    with torch.no_grad():
        for inputs_test, labels_test in test_dataloader:
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
            outputs_test = model(inputs_test)
            loss_test = criterion_test(outputs_test, labels_test)
            test_loss += loss_test.item()

            _, predicted_test = torch.max(outputs_test, 1)
            test_total += labels_test.size(0)
            test_correct += (predicted_test == labels_test).sum().item()

    test_accuracy = test_correct / test_total
    print(f'Test Loss: {test_loss / len(test_dataloader)}, Test Accuracy: {test_accuracy}')

def calculate_prediction_time(model, test_dataloader):
    total_time = 0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += end_time - start_time
            num_batches += 1

    print("Total prediction time for the entire test set:", total_time)
    print("Average prediction time per batch:", total_time / num_batches)

def calculate_model_size(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_bytes = num_params * 4
    model_size_mb = model_size_bytes / (1024 * 1024)
    return model_size_mb

def plot_confusion_matrix(model, test_dataloader, class_labels):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for inputs_test, labels_test in test_dataloader:
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
            outputs_test = model(inputs_test)
            output = (torch.max(torch.exp(outputs_test), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)
            labels = labels_test.data.cpu().numpy()
            y_true.extend(labels)

    y_true_names = [class_labels[i] for i in y_true]
    y_pred_names = [class_labels[i] for i in y_pred]
        
    cf_matrix = confusion_matrix(y_true_names, y_pred_names)
    plt.figure(figsize=(12, 7))
    plt.rcParams['font.family'] = 'Times New Roman'
    sn.heatmap(cf_matrix, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues_r")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def main():
    args = parse_args()

    model = load_model(args)

    test_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load dataset based on argument
    test_dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = ImageFolder(root=test_dataset_dir, transform=test_data_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    evaluate_model(model, test_dataloader)
    # calculate_prediction_time(model, test_dataloader)

    # model_size = calculate_model_size(model)
    # print("Model size:", model_size, "MB")

    # class_labels = ["Crazing", "Inclusion", "Patches", "Pitted Surface", "Rolled-in-Scale", "Scratches"]
    # plot_confusion_matrix(model, test_dataloader, class_labels)

if __name__ == "__main__":
    main()