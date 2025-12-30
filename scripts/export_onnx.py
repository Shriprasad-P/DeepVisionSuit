"""
ONNX Export Utility for DeepVision
===================================
Exports the TinyCNN model to ONNX format for cross-platform inference.
Optionally converts to TensorRT for NVIDIA GPU acceleration.

Usage:
    python scripts/export_onnx.py [--model_path PATH] [--output_path PATH]
"""

import torch
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import TinyCNN


def export_to_onnx(
    model_path: str = "best_tinycnn.pth",
    output_path: str = "model.onnx",
    num_classes: int = 200,
    opset_version: int = 17,
    dynamic_batch: bool = True
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to the trained .pth file
        output_path: Output ONNX file path
        num_classes: Number of output classes
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
    """
    print(f"📦 Exporting model to ONNX...")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_path}")
    
    # Load model
    model = TinyCNN(num_classes=num_classes)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        print(f"   ✓ Loaded weights from {model_path}")
    else:
        print(f"   ⚠ Weights not found, exporting untrained model")
    
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=64, width=64)
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    
    # Export using legacy exporter for compatibility
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # Use legacy exporter for compatibility
    )
    
    print(f"   ✓ Exported to {output_path}")
    
    # Verify export
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"   ✓ ONNX model validation passed")
    except ImportError:
        print(f"   ⚠ Install 'onnx' to validate the exported model")
    except Exception as e:
        print(f"   ✗ ONNX validation failed: {e}")
    
    # Model size
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"   📊 Model size: {size_mb:.2f} MB")
    
    return output_path


def test_onnx_inference(onnx_path: str):
    """Test inference with the exported ONNX model."""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print(f"\n🧪 Testing ONNX inference...")
        
        # Create session
        session = ort.InferenceSession(onnx_path)
        
        # Dummy input
        dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {"image": dummy_input})
        
        print(f"   ✓ Inference successful")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Predicted class: {np.argmax(outputs[0])}")
        
    except ImportError:
        print(f"   ⚠ Install 'onnxruntime' to test inference")


def export_to_tensorrt(onnx_path: str, trt_path: str = "model.trt"):
    """
    Convert ONNX model to TensorRT for NVIDIA GPU acceleration.
    Requires TensorRT to be installed.
    """
    try:
        import tensorrt as trt
        
        print(f"\n🚀 Converting to TensorRT...")
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        engine = builder.build_engine(network, config)
        
        # Serialize
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"   ✓ TensorRT engine saved to {trt_path}")
        return trt_path
        
    except ImportError:
        print(f"   ⚠ TensorRT not available. Install with: pip install tensorrt")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TinyCNN to ONNX/TensorRT")
    parser.add_argument("--model_path", type=str, default="best_tinycnn.pth",
                        help="Path to trained model weights")
    parser.add_argument("--output_path", type=str, default="model.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--tensorrt", action="store_true",
                        help="Also export to TensorRT")
    parser.add_argument("--test", action="store_true",
                        help="Test inference after export")
    
    args = parser.parse_args()
    
    # Export to ONNX
    onnx_path = export_to_onnx(args.model_path, args.output_path)
    
    # Test inference
    if args.test:
        test_onnx_inference(onnx_path)
    
    # Export to TensorRT
    if args.tensorrt:
        export_to_tensorrt(onnx_path)
    
    print("\n✅ Export complete!")
