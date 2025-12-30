import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
import time
import copy
import os
import argparse

# Weights & Biases (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not installed. Run 'pip install wandb' to enable experiment tracking.")

# Set Random Seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64 # Reduced for local run safety
NUM_EPOCHS = 5  # Reduced for demonstration
LEARNING_RATE = 1e-3
NUM_CLASSES = 200
IMAGE_SIZE = 64

# 1. Data Loading
print("Loading Tiny-ImageNet dataset (this may take a while first time)...")
# Using a smaller subset or streaming could be faster for testing, but let's try full load
try:
    dataset = load_dataset("zh-plus/tiny-imagenet")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have internet connection and 'datasets' installed.")
    exit(1)

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_train(examples):
    examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def preprocess_val(examples):
    examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

print("Preprocessing dataset...")
dataset['train'].set_transform(preprocess_train)
dataset['valid'].set_transform(preprocess_val)

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return pixel_values, labels

train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0) # num_workers=0 for safety on some systems
val_loader = DataLoader(dataset['valid'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

# 2. Model (CNN)
class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class TinyCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(SimpleResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(SimpleResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 3. Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, use_wandb=False):
    model.to(device)
    best_acc = 0.0
    
    # Initialize W&B if enabled
    if use_wandb and WANDB_AVAILABLE:
        wandb.watch(model, log='all', log_freq=100)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log to W&B
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_tinycnn.pth')
            
            # Log model artifact to W&B
            if use_wandb and WANDB_AVAILABLE:
                artifact = wandb.Artifact(
                    name='tinycnn-model',
                    type='model',
                    description=f'Best model with val_acc={val_acc:.2f}%'
                )
                artifact.add_file('best_tinycnn.pth')
                wandb.log_artifact(artifact)
            
        if scheduler:
            scheduler.step()
            
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    return model

# Run Training
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TinyCNN on Tiny-ImageNet')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='deepvision-tinycnn', help='W&B project name')
    args = parser.parse_args()
    
    # Initialize W&B
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                "architecture": "TinyCNN",
                "dataset": "Tiny-ImageNet",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealing",
            }
        )
        print(f"W&B run initialized: {wandb.run.name}")
    
    print("Initializing Model...")
    model = TinyCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print("Starting Training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=args.epochs, use_wandb=use_wandb)
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()
        print("W&B run completed.")
