"""
Unit Tests for DeepVision Model
================================
Tests model architecture, forward pass, and basic functionality.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import TinyCNN, SimpleResidualBlock


class TestSimpleResidualBlock:
    """Tests for the SimpleResidualBlock component."""
    
    def test_residual_block_same_channels(self):
        """Test residual block with same input/output channels."""
        block = SimpleResidualBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape
    
    def test_residual_block_channel_change(self):
        """Test residual block with different output channels."""
        block = SimpleResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)
    
    def test_residual_block_gradient_flow(self):
        """Test that gradients flow through residual connection."""
        block = SimpleResidualBlock(64, 64)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestTinyCNN:
    """Tests for the TinyCNN model."""
    
    def test_model_creation(self):
        """Test model can be instantiated."""
        model = TinyCNN(num_classes=200)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = TinyCNN(num_classes=200)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 200)
    
    def test_forward_pass_single_image(self):
        """Test forward pass with single image."""
        model = TinyCNN(num_classes=200)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 200)
    
    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [10, 100, 1000]:
            model = TinyCNN(num_classes=num_classes)
            x = torch.randn(2, 3, 64, 64)
            out = model(x)
            assert out.shape == (2, num_classes)
    
    def test_model_eval_mode(self):
        """Test model behavior in eval mode."""
        model = TinyCNN(num_classes=200)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(out1, out2)
    
    def test_parameter_count(self):
        """Test model has expected number of parameters."""
        model = TinyCNN(num_classes=200)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have ~11M parameters
        assert 10_000_000 < total_params < 15_000_000
    
    def test_gradient_computation(self):
        """Test gradients can be computed."""
        model = TinyCNN(num_classes=200)
        x = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 200, (2,))
        
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        
        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestModelIO:
    """Tests for model saving and loading."""
    
    def test_save_load_state_dict(self, tmp_path):
        """Test model can be saved and loaded."""
        model1 = TinyCNN(num_classes=200)
        save_path = tmp_path / "test_model.pth"
        
        # Save
        torch.save(model1.state_dict(), save_path)
        
        # Load
        model2 = TinyCNN(num_classes=200)
        model2.load_state_dict(torch.load(save_path, weights_only=True))
        
        # Compare outputs
        x = torch.randn(1, 3, 64, 64)
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2)


class TestInputValidation:
    """Tests for input edge cases."""
    
    def test_wrong_input_channels(self):
        """Test model fails gracefully with wrong input channels."""
        model = TinyCNN(num_classes=200)
        x = torch.randn(1, 1, 64, 64)  # Grayscale instead of RGB
        
        with pytest.raises(RuntimeError):
            model(x)
    
    def test_wrong_input_size(self):
        """Test model handles different input sizes."""
        model = TinyCNN(num_classes=200)
        
        # Due to AdaptiveAvgPool, different sizes should work
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
