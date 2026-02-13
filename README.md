# 🧠 DeepVision Suite


## Team Members: 
- **1. Nuthalapati sai Bharath Kumar**
- **2. Pranjal Prakash Pandey**
- **3. Shriprasad Patil**


A comprehensive deep learning image classification system built with PyTorch, featuring model interpretability tools and MLOps practices.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Overview

DeepVision Suite is a production-ready image classification system trained on the **Tiny-ImageNet** dataset (200 classes). It goes beyond just achieving high accuracy—it focuses on **Model Interpretability** and includes tools for understanding *why* the model makes certain decisions.

### Key Features
- 🏗️ **Custom ResNet Architecture** with residual connections
- 📊 **Model Visualization Suite** (Grad-CAM, Feature Maps, Filter Visualization)
- 🔄 **MLOps Integration** (W&B tracking, GitHub Actions CI/CD)
- 🚀 **Production Ready** (FastAPI serving, ONNX export)

## 📁 Project Structure

```
DeepVisionSuite/
├── train.py                 # Main training script
├── visualize_model.py       # Visualization utilities (Grad-CAM, etc.)
├── requirements.txt         # Python dependencies
├── configs/
│   └── train_config.yaml    # Training configuration
├── scripts/
│   └── export_onnx.py       # ONNX model export
├── serve/
│   ├── app.py               # FastAPI inference server
│   └── Dockerfile           # Container for deployment
├── tests/
│   └── test_model.py        # Unit tests
├── visualizations/          # Generated visualization outputs
└── .github/workflows/       # CI/CD pipelines
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Shriprasad-P/DeepVisionSuit.git
cd DeepVisionSuit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Training

### Basic Training
```bash
python train.py --epochs 10
```

### With Weights & Biases Tracking
```bash
python train.py --epochs 20 --wandb
```

### Configuration Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--wandb` | False | Enable W&B experiment tracking |

## 📊 Model Visualization

Generate interpretability visualizations:

```bash
python visualize_model.py
```

This creates:
- **Architecture Diagram** - Network structure visualization
- **Conv1 Filters** - Learned first-layer features
- **Feature Maps** - Layer-by-layer activations
- **Grad-CAM** - Class activation heatmaps

## 🚀 Deployment

### FastAPI Server
```bash
cd serve
uvicorn app:app --reload --port 8000
```

### Docker
```bash
cd serve
docker build -t deepvision-api .
docker run -p 8000:8000 deepvision-api
```

### ONNX Export
```bash
python scripts/export_onnx.py
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 🏗️ Architecture

The model uses a custom ResNet-style architecture with:
- **Residual Connections** - Prevents vanishing gradients
- **Batch Normalization** - Stabilizes training
- **AdamW Optimizer** - Weight decay regularization
- **Cosine Annealing LR** - Learning rate scheduling

## 📈 Results

Training on Tiny-ImageNet (200 classes, 64x64 images):
- Dataset: 100,000 training / 10,000 validation images
- Architecture: Custom ResNet with 4 residual blocks

## 🔮 Future Improvements

- [ ] Vision Transformer (ViT) implementation
- [ ] TensorRT optimization for edge deployment
- [ ] Object detection with YOLO backbone
- [ ] Medical imaging fine-tuning

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ using PyTorch
