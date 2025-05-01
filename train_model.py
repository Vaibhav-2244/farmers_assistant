import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import ssl    
import urllib.request
import certifi
import random
import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Handle SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, save_path):
    """Download a file with proper SSL context"""
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ssl_context) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        try:
            # Try loading ResNet50 with SSL verification
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True, trust_repo=True)
        except Exception as e:
            print(f"Error loading model from torch.hub: {e}")
            print("Attempting alternative model initialization...")
            
            # Alternative: Create ResNet50 without pre-trained weights
            from torchvision.models import resnet50
            self.model = resnet50()
            
            # Download weights manually if needed
            weights_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
            weights_path = os.path.join('models', 'resnet50-0676ba61.pth')
            
            if not os.path.exists(weights_path):
                os.makedirs('models', exist_ok=True)
                print("Downloading ResNet50 weights...")
                if download_file(weights_url, weights_path):
                    # Load the weights
                    state_dict = torch.load(weights_path)
                    self.model.load_state_dict(state_dict)
                    print("Successfully loaded model weights!")
                else:
                    print("Warning: Could not download pre-trained weights. Using random initialization.")
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def save_model_info(selected_classes, class_mapping, total_classes, model_path='models/model_info.json'):
    """Save information about the trained model and its categories"""
    os.makedirs('models', exist_ok=True)
    model_info = {
        "trained_categories": selected_classes,
        "class_mapping": class_mapping,
        "model_version": "1.0",
        "image_size": 160,
        "date_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "total_categories": len(total_classes),
            "used_categories": len(selected_classes),
            "percentage_used": 35
        }
    }
    
    with open(model_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    print(f"\nModel information saved to {model_path}")
    
    # Also save a human-readable summary
    with open('models/trained_categories_summary.txt', 'w') as f:
        f.write("Trained Categories Summary\n")
        f.write("========================\n\n")
        f.write(f"Total Categories in Dataset: {len(total_classes)}\n")
        f.write(f"Categories Used for Training: {len(selected_classes)}\n")
        f.write(f"Percentage Used: 35%\n\n")
        f.write("Selected Categories:\n")
        f.write("-------------------\n")
        for idx, category in enumerate(selected_classes, 1):
            f.write(f"{idx}. {category}\n")
    
    print("Category summary saved to models/trained_categories_summary.txt")

def check_category_supported(category_name, model_info_path='models/model_info.json'):
    """Check if a given plant category was part of the training"""
    try:
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        trained_categories = model_info["trained_categories"]
        is_supported = category_name in trained_categories
        
        if is_supported:
            print(f"✓ Category '{category_name}' is supported by the model.")
        else:
            print(f"✗ Category '{category_name}' is NOT supported by this model.")
            print("\nSupported categories are:")
            for idx, cat in enumerate(trained_categories, 1):
                print(f"{idx}. {cat}")
        
        return is_supported
    except FileNotFoundError:
        print("Error: Model information file not found. Please train the model first.")
        return False

def train_model():
    print("Checking for PlantVillage dataset...")
    if not os.path.exists('data/PlantVillage'):
        print("Error: PlantVillage dataset not found in data/PlantVillage")
        print("Please ensure your dataset is organized as follows:")
        print("data/PlantVillage/")
        print("├── Disease_1/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("│   └── ...")
        print("├── Disease_2/")
        print("└── ...")
        return

    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(160),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Just resize and normalize for validation
    val_transform = transforms.Compose([
        transforms.Resize(180),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Load dataset
        print("Loading dataset...")
        dataset_path = 'data/PlantVillage'
        full_dataset = ImageFolder(dataset_path, transform=train_transform)
        
        # Select 35% of the classes randomly
        all_classes = sorted(full_dataset.classes)
        num_classes = len(all_classes)
        num_selected = int(num_classes * 0.35)  # Select 35% of classes
        selected_classes = sorted(random.sample(all_classes, num_selected))
        
        print("\nSelected plant categories for training:")
        print("----------------------------------------")
        print(f"Using {len(selected_classes)} out of {num_classes} categories (35%)")
        print("\nSelected categories:")
        for idx, class_name in enumerate(selected_classes, 1):
            print(f"{idx}. {class_name}")
        
        # Create a mask for selected classes
        selected_indices = []
        for idx, (path, class_idx) in enumerate(full_dataset.samples):
            if full_dataset.classes[class_idx] in selected_classes:
                selected_indices.append(idx)
        
        # Create a subset with only selected classes
        reduced_dataset = Subset(full_dataset, selected_indices)
        
        # Update class mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Save selected classes information
        os.makedirs('data', exist_ok=True)
        with open('data/class_indices.json', 'w') as f:
            json.dump({
                "selected_classes": selected_classes,
                "class_mapping": idx_to_class
            }, f, indent=4)
        
        # Split dataset
        print("\nSplitting dataset into train and validation sets...")
        train_size = int(0.8 * len(reduced_dataset))
        val_size = len(reduced_dataset) - train_size
        train_dataset, val_dataset = random_split(reduced_dataset, [train_size, val_size])
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        # Initialize model with reduced number of classes
        print("\nInitializing model...")
        num_classes = len(selected_classes)
        model = PlantDiseaseModel(num_classes)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
        
        # Training loop
        print("Starting training...")
        num_epochs = 5  # Reduced to 5 epochs
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for inputs, labels in train_bar:
                # Remap labels to new indices
                remapped_labels = torch.tensor([class_to_idx[full_dataset.classes[label]] for label in labels])
                inputs, remapped_labels = inputs.to(device), remapped_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, remapped_labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                train_bar.set_postfix({'loss': loss.item()})
            
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Remap labels to new indices
                    remapped_labels = torch.tensor([class_to_idx[full_dataset.classes[label]] for label in labels])
                    inputs, remapped_labels = inputs.to(device), remapped_labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, remapped_labels)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += remapped_labels.size(0)
                    correct += predicted.eq(remapped_labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
            accuracy = 100. * correct / total
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Accuracy: {accuracy:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'idx_to_class': idx_to_class,
                    'selected_classes': selected_classes
                }, 'models/plant_disease_model.pth')
                print('Saved best model!')
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_curves.png')
        plt.close()
        
        # After successful training, save model info
        save_model_info(selected_classes, idx_to_class, all_classes)
        
        print("\nTraining completed successfully!")
        print("To check if a specific plant category is supported, use:")
        print("check_category_supported('category_name')")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

if __name__ == '__main__':
    try:
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        print("Starting model training...")
        train_model()
        
        # Example usage of category checking
        print("\nExample: Checking if 'Tomato_Early_blight' is supported:")
        check_category_supported('Tomato_Early_blight')
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Please check the error message above and ensure all requirements are met.") 