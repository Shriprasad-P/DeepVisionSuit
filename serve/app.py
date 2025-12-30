"""
FastAPI Inference Server for DeepVision
========================================
Provides REST API endpoints for image classification using the trained TinyCNN model.

Run with: uvicorn serve.app:app --reload --port 8000
API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import TinyCNN

# Global model variable
model = None
device = None
transform = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, device, transform
    
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"🔧 Loading model on device: {device}")
    
    # Load model
    model = TinyCNN(num_classes=200)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_tinycnn.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"⚠ Model file not found at {model_path}. Using untrained model.")
    
    model.to(device)
    model.eval()
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("🚀 Model ready for inference!")
    yield
    
    # Cleanup
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="DeepVision API",
    description="Image classification API using TinyCNN trained on Tiny-ImageNet",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "DeepVision API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with predicted class ID, confidence, and top-5 predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Top-5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Build response
        top5_predictions = [
            {"class_id": int(idx), "confidence": float(prob)}
            for idx, prob in zip(top5_indices[0], top5_probs[0])
        ]
        
        return {
            "success": True,
            "prediction": {
                "class_id": top5_predictions[0]["class_id"],
                "confidence": top5_predictions[0]["confidence"]
            },
            "top5": top5_predictions,
            "inference_time_ms": round(inference_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model architecture information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "architecture": "TinyCNN",
        "num_classes": 200,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(total_params * 4 / 1024 / 1024, 2),
        "device": str(device)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
