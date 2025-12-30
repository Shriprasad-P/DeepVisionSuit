"""
Model Visualization Suite for TinyCNN
======================================
Provides comprehensive visualizations for the trained TinyCNN model:
1. Architecture Diagram
2. Feature Map Activations
3. Convolutional Filter Visualization
4. Grad-CAM Attention Maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from PIL import Image
import os
from tqdm import tqdm

# Import the model from train.py
from train import TinyCNN, SimpleResidualBlock

# Device Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# Create output directory for visualizations
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_path='best_tinycnn.pth'):
    """Load the trained model."""
    model = TinyCNN(num_classes=200)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"⚠ Model file {model_path} not found. Using untrained model.")
    model.to(device)
    model.eval()
    return model


def get_sample_images(num_samples=5):
    """Load sample images from the dataset."""
    print("Loading sample images from Tiny-ImageNet...")
    dataset = load_dataset("zh-plus/tiny-imagenet", split="valid")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    original_images = []
    labels = []
    
    for i in range(min(num_samples, len(dataset))):
        img = dataset[i]['image'].convert('RGB')
        original_images.append(img)
        images.append(transform(img))
        labels.append(dataset[i]['label'])
    
    return torch.stack(images), original_images, labels


# ==========================================
# 1. Architecture Visualization
# ==========================================

def visualize_architecture():
    """Create a visual diagram of the TinyCNN architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Color scheme
    colors = {
        'input': '#e94560',
        'conv': '#0f3460',
        'residual': '#16213e',
        'pool': '#533483',
        'fc': '#e94560',
        'output': '#00d9ff'
    }
    
    # Layer definitions with positions
    layers = [
        {'name': 'Input\n64×64×3', 'x': 1, 'color': colors['input'], 'size': (1.2, 2.5)},
        {'name': 'Conv1\n64×64×64\n3×3, BN, ReLU', 'x': 3, 'color': colors['conv'], 'size': (1.4, 3)},
        {'name': 'Layer1\n64×64×64\n2 ResBlocks', 'x': 5.5, 'color': colors['residual'], 'size': (1.6, 3.5)},
        {'name': 'Layer2\n32×32×128\n2 ResBlocks', 'x': 8, 'color': colors['residual'], 'size': (1.6, 3.2)},
        {'name': 'Layer3\n16×16×256\n2 ResBlocks', 'x': 10.5, 'color': colors['residual'], 'size': (1.6, 2.8)},
        {'name': 'Layer4\n8×8×512\n2 ResBlocks', 'x': 13, 'color': colors['residual'], 'size': (1.6, 2.5)},
        {'name': 'AdaptiveAvgPool\n1×1×512', 'x': 15.5, 'color': colors['pool'], 'size': (1.4, 2)},
        {'name': 'Flatten\n512', 'x': 17.5, 'color': colors['pool'], 'size': (1.2, 1.5)},
        {'name': 'FC\n512→200', 'x': 19.5, 'color': colors['fc'], 'size': (1.2, 2)},
        {'name': 'Output\n200 classes', 'x': 21.5, 'color': colors['output'], 'size': (1.2, 2.5)},
    ]
    
    y_center = 5
    
    # Draw layers
    for layer in layers:
        width, height = layer['size']
        rect = FancyBboxPatch(
            (layer['x'] - width/2, y_center - height/2),
            width, height,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=layer['color'],
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(layer['x'], y_center, layer['name'], 
                ha='center', va='center', color='white', 
                fontsize=8, fontweight='bold', wrap=True)
    
    # Draw arrows between layers
    for i in range(len(layers) - 1):
        x1 = layers[i]['x'] + layers[i]['size'][0]/2
        x2 = layers[i+1]['x'] - layers[i+1]['size'][0]/2
        ax.annotate('', xy=(x2 - 0.1, y_center), xytext=(x1 + 0.1, y_center),
                    arrowprops=dict(arrowstyle='->', color='#00d9ff', lw=2))
    
    # Title
    ax.text(11, 9, 'TinyCNN Architecture (ResNet-style)', 
            ha='center', va='center', color='white', 
            fontsize=18, fontweight='bold')
    
    # Residual Block detail
    ax.text(11, 1.5, 'Residual Block: Conv3×3 → BN → ReLU → Conv3×3 → BN → (+Skip) → ReLU', 
            ha='center', va='center', color='#00d9ff', 
            fontsize=10, style='italic')
    
    # Parameter count
    model = TinyCNN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ax.text(11, 0.7, f'Total Parameters: {total_params:,} | Trainable: {trainable_params:,}', 
            ha='center', va='center', color='#aaa', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/architecture_diagram.png', dpi=150, 
                facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved architecture diagram to {OUTPUT_DIR}/architecture_diagram.png")


# ==========================================
# 2. Feature Map Visualization
# ==========================================

class FeatureExtractor:
    """Extract intermediate feature maps from the model."""
    def __init__(self, model):
        self.model = model
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        self.model.conv1.register_forward_hook(hook_fn('conv1'))
        self.model.layer1.register_forward_hook(hook_fn('layer1'))
        self.model.layer2.register_forward_hook(hook_fn('layer2'))
        self.model.layer3.register_forward_hook(hook_fn('layer3'))
        self.model.layer4.register_forward_hook(hook_fn('layer4'))
    
    def __call__(self, x):
        with torch.no_grad():
            self.model(x.to(device))
        return self.features


def visualize_feature_maps(model, image_tensor, original_image, sample_idx=0):
    """Visualize feature maps at different layers."""
    extractor = FeatureExtractor(model)
    features = extractor(image_tensor.unsqueeze(0))
    
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Title
    fig.suptitle('Feature Map Activations Across Layers', 
                 fontsize=20, fontweight='bold', color='white', y=0.98)
    
    # Original image
    ax_orig = fig.add_subplot(2, 5, 1)
    ax_orig.imshow(original_image)
    ax_orig.set_title('Original Image', color='white', fontsize=12)
    ax_orig.axis('off')
    
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    positions = [2, 3, 4, 5, 7]  # Grid positions
    
    for idx, (name, pos) in enumerate(zip(layer_names, positions)):
        ax = fig.add_subplot(2, 5, pos)
        feat = features[name][0].cpu()
        
        # Show average of feature maps
        avg_feat = feat.mean(dim=0).numpy()
        im = ax.imshow(avg_feat, cmap='viridis')
        ax.set_title(f'{name}\n{feat.shape[0]} channels', color='white', fontsize=10)
        ax.axis('off')
    
    # Show individual filters from layer4
    ax_filters = fig.add_subplot(2, 5, 8)
    layer4_feat = features['layer4'][0].cpu()
    # Create a grid of first 16 feature maps
    n_show = min(16, layer4_feat.shape[0])
    grid_size = int(np.ceil(np.sqrt(n_show)))
    
    grid = np.zeros((grid_size * layer4_feat.shape[1], grid_size * layer4_feat.shape[2]))
    for i in range(n_show):
        row = i // grid_size
        col = i % grid_size
        grid[row*layer4_feat.shape[1]:(row+1)*layer4_feat.shape[1],
             col*layer4_feat.shape[2]:(col+1)*layer4_feat.shape[2]] = layer4_feat[i].numpy()
    
    ax_filters.imshow(grid, cmap='plasma')
    ax_filters.set_title('Layer4 Feature Grid\n(First 16 channels)', color='white', fontsize=10)
    ax_filters.axis('off')
    
    # Feature map statistics
    ax_stats = fig.add_subplot(2, 5, 9)
    ax_stats.set_facecolor('#1a1a2e')
    stats_text = "Feature Map Statistics:\n\n"
    for name in layer_names:
        feat = features[name][0]
        stats_text += f"{name}: {tuple(feat.shape)}\n"
        stats_text += f"  Mean: {feat.mean():.3f}, Std: {feat.std():.3f}\n\n"
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, color='white', verticalalignment='top', 
                  fontfamily='monospace')
    ax_stats.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_maps_{sample_idx}.png', dpi=150,
                facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved feature maps to {OUTPUT_DIR}/feature_maps_{sample_idx}.png")


# ==========================================
# 3. Filter Visualization
# ==========================================

def visualize_filters(model):
    """Visualize convolutional filters from the first layer."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('First Layer Convolutional Filters (3→64)', 
                 fontsize=18, fontweight='bold', color='white', y=0.98)
    
    # Get first conv layer weights
    conv1_weights = model.conv1.weight.detach().cpu()
    n_filters = conv1_weights.shape[0]
    
    # Normalize for visualization
    weights_min = conv1_weights.min()
    weights_max = conv1_weights.max()
    conv1_weights = (conv1_weights - weights_min) / (weights_max - weights_min)
    
    # Create grid
    rows, cols = 8, 8
    for i in range(min(64, n_filters)):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Get filter and transpose for RGB display
        filter_img = conv1_weights[i].permute(1, 2, 0).numpy()
        ax.imshow(filter_img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/conv1_filters.png', dpi=150,
                facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved filter visualization to {OUTPUT_DIR}/conv1_filters.png")


# ==========================================
# 4. Grad-CAM Visualization
# ==========================================

class GradCAM:
    """Grad-CAM implementation for visualizing model attention."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_image, target_class=None):
        self.model.eval()
        input_image = input_image.unsqueeze(0).to(device)
        input_image.requires_grad = True
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu()
        activations = self.activations[0].cpu()
        
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.numpy(), target_class, output[0, target_class].item()


def visualize_gradcam(model, images, original_images, labels, num_samples=5):
    """Generate Grad-CAM visualizations for multiple images."""
    gradcam = GradCAM(model, model.layer4)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Grad-CAM: What the Model Focuses On', 
                 fontsize=18, fontweight='bold', color='white', y=0.98)
    
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(original_images[i])
        axes[0, i].set_title(f'Label: {labels[i]}', color='white', fontsize=10)
        axes[0, i].axis('off')
        
        # Grad-CAM heatmap
        cam, pred_class, confidence = gradcam.generate(images[i])
        cam_resized = np.array(Image.fromarray(cam).resize((64, 64), Image.BILINEAR))
        
        axes[1, i].imshow(cam_resized, cmap='jet')
        axes[1, i].set_title(f'Pred: {pred_class}\nConf: {confidence:.2f}', 
                             color='white', fontsize=10)
        axes[1, i].axis('off')
        
        # Overlay
        original_np = np.array(original_images[i].resize((64, 64)))
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = (0.6 * original_np / 255.0 + 0.4 * heatmap)
        overlay = np.clip(overlay, 0, 1)
        
        axes[2, i].imshow(overlay)
        axes[2, i].set_title('Overlay', color='white', fontsize=10)
        axes[2, i].axis('off')
    
    # Row labels
    axes[0, 0].text(-0.2, 0.5, 'Original', transform=axes[0, 0].transAxes,
                    fontsize=12, color='white', rotation=90, va='center')
    axes[1, 0].text(-0.2, 0.5, 'Heatmap', transform=axes[1, 0].transAxes,
                    fontsize=12, color='white', rotation=90, va='center')
    axes[2, 0].text(-0.2, 0.5, 'Overlay', transform=axes[2, 0].transAxes,
                    fontsize=12, color='white', rotation=90, va='center')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gradcam_visualization.png', dpi=150,
                facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Grad-CAM visualization to {OUTPUT_DIR}/gradcam_visualization.png")


# ==========================================
# 5. Model Summary
# ==========================================

def print_model_summary(model):
    """Print a detailed model summary."""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    # Count parameters per layer
    total_params = 0
    trainable_params = 0
    
    layer_info = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += params
            trainable_params += trainable
            layer_info.append((name, type(module).__name__, params))
    
    # Print table
    print(f"{'Layer Name':<30} {'Type':<20} {'Parameters':>15}")
    print("-"*65)
    for name, type_name, params in layer_info:
        print(f"{name:<30} {type_name:<20} {params:>15,}")
    
    print("-"*65)
    print(f"{'Total Parameters:':<50} {total_params:>15,}")
    print(f"{'Trainable Parameters:':<50} {trainable_params:>15,}")
    print(f"{'Model Size (MB):':<50} {total_params * 4 / 1024 / 1024:>15.2f}")
    print("="*60 + "\n")


# ==========================================
# Main Execution
# ==========================================

def main():
    print("\n" + "="*60)
    print("🔬 TinyCNN Visualization Suite")
    print("="*60 + "\n")
    
    # Load model
    model = load_model()
    
    # Print model summary
    print_model_summary(model)
    
    # Generate visualizations
    print("\n📊 Generating Visualizations...\n")
    
    # 1. Architecture diagram
    print("1. Creating architecture diagram...")
    visualize_architecture()
    
    # 2. Load sample images
    print("\n2. Loading sample images...")
    images, original_images, labels = get_sample_images(num_samples=5)
    
    # 3. Feature maps
    print("\n3. Generating feature map visualizations...")
    visualize_feature_maps(model, images[0], original_images[0], sample_idx=0)
    
    # 4. Filter visualization
    print("\n4. Visualizing convolutional filters...")
    visualize_filters(model)
    
    # 5. Grad-CAM
    print("\n5. Generating Grad-CAM visualizations...")
    visualize_gradcam(model, images, original_images, labels)
    
    print("\n" + "="*60)
    print("✅ All visualizations saved to:", OUTPUT_DIR)
    print("="*60 + "\n")
    
    # List generated files
    print("Generated files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  📁 {OUTPUT_DIR}/{f}")


if __name__ == "__main__":
    main()
