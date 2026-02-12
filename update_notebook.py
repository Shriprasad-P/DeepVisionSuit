
import json

# Load the notebook
notebook_path = 'TinyImageNet_Project.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to create cells
def new_markdown_cell(source):
    # Split by newlines and add \n to each line for ipynb format
    lines = [line + '\n' for line in source.split('\n')]
    if lines and lines[-1] == '\n':
        lines[-1] = '' # Remove last empty newline if present
    # Better: just split and ensure logical lines
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + '\n' for l in source.split('\n')]
    }

def new_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + '\n' for l in source.split('\n')]
    }

# Define the new content
viz_imports_code = """# Visualization Imports
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from PIL import Image"""

viz_arch_code = """# ==========================================
# 1. Architecture Visualization
# ==========================================

def visualize_architecture():
    \"\"\"Create a visual diagram of the TinyCNN architecture.\"\"\"
    try:
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
            {'name': 'Input\\n64×64×3', 'x': 1, 'color': colors['input'], 'size': (1.2, 2.5)},
            {'name': 'Conv1\\n64×64×64\\n3×3, BN, ReLU', 'x': 3, 'color': colors['conv'], 'size': (1.4, 3)},
            {'name': 'Layer1\\n64×64×64\\n2 ResBlocks', 'x': 5.5, 'color': colors['residual'], 'size': (1.6, 3.5)},
            {'name': 'Layer2\\n32×32×128\\n2 ResBlocks', 'x': 8, 'color': colors['residual'], 'size': (1.6, 3.2)},
            {'name': 'Layer3\\n16×16×256\\n2 ResBlocks', 'x': 10.5, 'color': colors['residual'], 'size': (1.6, 2.8)},
            {'name': 'Layer4\\n8×8×512\\n2 ResBlocks', 'x': 13, 'color': colors['residual'], 'size': (1.6, 2.5)},
            {'name': 'AdaptiveAvgPool\\n1×1×512', 'x': 15.5, 'color': colors['pool'], 'size': (1.4, 2)},
            {'name': 'Flatten\\n512', 'x': 17.5, 'color': colors['pool'], 'size': (1.2, 1.5)},
            {'name': 'FC\\n512→200', 'x': 19.5, 'color': colors['fc'], 'size': (1.2, 2)},
            {'name': 'Output\\n200 classes', 'x': 21.5, 'color': colors['output'], 'size': (1.2, 2.5)},
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
        # Assuming TinyCNN class is available in the notebook
        temp_model = TinyCNN() 
        total_params = sum(p.numel() for p in temp_model.parameters())
        trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        ax.text(11, 0.7, f'Total Parameters: {total_params:,} | Trainable: {trainable_params:,}', 
                ha='center', va='center', color='#aaa', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization Error: {e}")"""

viz_features_code = """# ==========================================
# 2. Feature Map Visualization
# ==========================================

class FeatureExtractor:
    \"\"\"Extract intermediate feature maps from the model.\"\"\"
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        # Check if model has these attributes
        layers_to_hook = {
            'conv1': self.model.conv1,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4
        }

        for name, layer in layers_to_hook.items():
             self.hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self, x):
        self.features = {} # Clear previous
        with torch.no_grad():
            self.model(x.to(device))
        return self.features


def visualize_feature_maps(model, image_tensor, original_image, sample_idx=0):
    \"\"\"Visualize feature maps at different layers.\"\"\"
    extractor = FeatureExtractor(model)
    try:
        features = extractor(image_tensor.unsqueeze(0))
        
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Title
        fig.suptitle('Feature Map Activations Across Layers', 
                     fontsize=20, fontweight='bold', color='white', y=0.98)
        
        # Original image (Handling PIL vs Tensor)
        ax_orig = fig.add_subplot(2, 5, 1)
        if isinstance(original_image, torch.Tensor):
             # Un-normalize if it's a tensor
             img_disp = original_image.permute(1, 2, 0).cpu().numpy()
             mean = np.array([0.485, 0.456, 0.406])
             std = np.array([0.229, 0.224, 0.225])
             img_disp = std * img_disp + mean
             img_disp = np.clip(img_disp, 0, 1)
             ax_orig.imshow(img_disp)
        else:
            ax_orig.imshow(original_image)

        ax_orig.set_title('Original Image', color='white', fontsize=12)
        ax_orig.axis('off')
        
        layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        positions = [2, 3, 4, 5, 7]  # Grid positions
        
        for idx, (name, pos) in enumerate(zip(layer_names, positions)):
            if name in features:
                ax = fig.add_subplot(2, 5, pos)
                feat = features[name][0].cpu()
                
                # Show average of feature maps
                avg_feat = feat.mean(dim=0).numpy()
                im = ax.imshow(avg_feat, cmap='viridis')
                ax.set_title(f'{name}\\n{feat.shape[0]} channels', color='white', fontsize=10)
                ax.axis('off')
        
        # Show individual filters from layer4
        if 'layer4' in features:
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
            ax_filters.set_title('Layer4 Feature Grid\\n(First 16 channels)', color='white', fontsize=10)
            ax_filters.axis('off')
        
        plt.tight_layout()
        plt.show()
    finally:
        extractor.remove_hooks()"""

viz_filters_code = """# ==========================================
# 3. Filter Visualization
# ==========================================

def visualize_filters(model):
    \"\"\"Visualize convolutional filters from the first layer.\"\"\"
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
    plt.show()"""

viz_gradcam_code = """# ==========================================
# 4. Grad-CAM Visualization
# ==========================================

class GradCAM:
    \"\"\"Grad-CAM implementation for visualizing model attention.\"\"\"
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
    
    def generate(self, input_image, target_class=None):
        self.model.eval()
        # input_image should be [C, H, W]
        # Ensure we have a batch dim
        if input_image.dim() == 3:
            input_tensor = input_image.unsqueeze(0).to(device)
        else:
            input_tensor = input_image.to(device)
            
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
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


def visualize_gradcam(model, dataset, num_samples=5):
    \"\"\"Generate Grad-CAM visualizations for multiple images from the validation set.\"\"\"
    # We need to find the last convolutional layer. In TinyCNN it is model.layer4[-1] or similar.
    # But layer4 is a Sequential, so we should attach to the last block or just layer4 output?
    # visualize_model.py used model.layer4 so we will stick to that.
    
    gradcam = GradCAM(model, model.layer4)
    
    try:
        # Get random samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle('Grad-CAM: What the Model Focuses On', 
                     fontsize=18, fontweight='bold', color='white', y=0.98)
        
        for col, idx in enumerate(indices):
             # Get item
             # Note: dataset setup in notebook might differ slightly from script
             # In notebook: dataset['valid'] and transforms are applied via set_transform or manually?
             # Cell 78: dataset['valid'].set_transform(preprocess_val)
             # So indexing gives us pixel_values directly?
             
             # The huggingface dataset with set_transform returns the transformed item
             item = dataset[idx] # keys: image, label, pixel_values
             
             # pixel_values is a tensor [3, 64, 64]
             input_tensor = item['pixel_values']
             label = item['label']
             
             # Original image for display - huggingface dataset stores PIL image in 'image' key if we access raw
             # But set_transform might override or hide it depending on how it's implemented.
             # Actually preprocess_val returns dict with pixel_values.
             # Let's check if we can get the original image easily. 
             # We can't easily get the original PIL image if set_transform is active returning only pixel_values
             # WE will un-normalize the tensor for "original" display.
             
             original_np = input_tensor.permute(1, 2, 0).numpy()
             mean = np.array([0.485, 0.456, 0.406])
             std = np.array([0.229, 0.224, 0.225])
             original_np = std * original_np + mean
             original_np = np.clip(original_np, 0, 1) # 0-1 float
             
             # Grad-CAM heatmap
             cam, pred_class, confidence = gradcam.generate(input_tensor)
             cam_resized = np.array(Image.fromarray(cam).resize((64, 64), Image.BILINEAR))
             
             # Top row: Original
             axes[0, col].imshow(original_np)
             axes[0, col].set_title(f'Label: {label}', color='white', fontsize=10)
             axes[0, col].axis('off')
             
             # Middle row: Heatmap
             axes[1, col].imshow(cam_resized, cmap='jet')
             axes[1, col].set_title(f'Pred: {pred_class}\\nConf: {confidence:.2f}', 
                                  color='white', fontsize=10)
             axes[1, col].axis('off')
             
             # Bottom row: Overlay
             heatmap = plt.cm.jet(cam_resized)[:, :, :3]
             overlay = (0.6 * original_np + 0.4 * heatmap)
             overlay = np.clip(overlay, 0, 1)
             
             axes[2, col].imshow(overlay)
             axes[2, col].set_title('Overlay', color='white', fontsize=10)
             axes[2, col].axis('off')
        
        # Row labels
        if num_samples > 0:
            axes[0, 0].text(-0.2, 0.5, 'Original', transform=axes[0, 0].transAxes,
                            fontsize=12, color='white', rotation=90, va='center')
            axes[1, 0].text(-0.2, 0.5, 'Heatmap', transform=axes[1, 0].transAxes,
                            fontsize=12, color='white', rotation=90, va='center')
            axes[2, 0].text(-0.2, 0.5, 'Overlay', transform=axes[2, 0].transAxes,
                            fontsize=12, color='white', rotation=90, va='center')
        
        plt.tight_layout()
        plt.show()
    finally:
        gradcam.remove_hooks()"""

viz_run_code = """# Run Visualizations
print("Generating Visualizations for CNN Model...")

# 1. Architecture
print("\\n1. Architecture Diagram")
visualize_architecture()

# 2. Advanced Visualizations using Validation Data
print("\\n2. Advanced Analysis (Feature Maps, Filters, Grad-CAM)")
# We use the 'dataset' object from earlier cells which has 'valid' split
# check if cnn_model is defined (it should be after Cell 453)
if 'cnn_model' in globals():
    # Get a single sample for feature maps
    sample_item = dataset['valid'][0]
    sample_tensor = sample_item['pixel_values']
    # Un-normalize for 'original' arg
    # Note: visualize_feature_maps handles tensor input now
    
    print("Visualizing Feature Maps...")
    visualize_feature_maps(cnn_model, sample_tensor, sample_tensor)
    
    print("Visualizing Filters...")
    visualize_filters(cnn_model)
    
    print("Visualizing Grad-CAM...")
    visualize_gradcam(cnn_model, dataset['valid'], num_samples=5)
else:
    print("CNN Model not found. Please run the training cells above.")"""

# Creating cells
cell_md_header = new_markdown_cell("## 5. Advanced Model Visualization")
cell_imports = new_code_cell(viz_imports_code)
cell_arch = new_code_cell(viz_arch_code)
cell_features = new_code_cell(viz_features_code)
cell_filters = new_code_cell(viz_filters_code)
cell_gradcam = new_code_cell(viz_gradcam_code)
cell_run = new_code_cell(viz_run_code)

# Append to notebook
nb['cells'].extend([
    new_markdown_cell("## 5. Advanced Model Visualization"),
    new_code_cell(viz_imports_code),
    new_code_cell(viz_arch_code),
    new_code_cell(viz_features_code),
    new_code_cell(viz_filters_code),
    new_code_cell(viz_gradcam_code),
    new_code_cell(viz_run_code)
])

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Updated {notebook_path} with visualization cells.")
