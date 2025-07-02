import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = OmegaConf.create({
        "model": {
            "type": "test_model",
            "params": {
                "in_channels": 3,
                "out_channels": 3,
                "hidden_dim": 128
            }
        },
        "data": {
            "batch_size": 4,
            "num_workers": 0,
            "dataset": "test_dataset"
        },
        "training": {
            "epochs": 10,
            "learning_rate": 1e-4,
            "optimizer": "adam"
        },
        "paths": {
            "data_dir": "/tmp/test_data",
            "output_dir": "/tmp/test_output"
        }
    })
    return config


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def sample_video_tensor():
    """Create a sample video tensor for testing (batch, time, channels, height, width)."""
    return torch.randn(2, 8, 3, 64, 64)


@pytest.fixture
def mock_dataset_item():
    """Create a mock dataset item."""
    return {
        "image": torch.randn(3, 256, 256),
        "label": torch.randint(0, 10, (1,)).item(),
        "metadata": {
            "filename": "test_image.jpg",
            "timestamp": 1234567890
        }
    }


@pytest.fixture
def mock_model_state():
    """Create a mock model state dictionary."""
    return {
        "encoder.weight": torch.randn(128, 3, 3, 3),
        "encoder.bias": torch.randn(128),
        "decoder.weight": torch.randn(3, 128, 3, 3),
        "decoder.bias": torch.randn(3),
        "epoch": 5,
        "global_step": 1000,
        "optimizer_state": {}
    }


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_checkpoint_path(temp_dir):
    """Create a mock checkpoint path."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir / "checkpoint_epoch_5.pt"


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb for testing without actual logging."""
    class MockWandb:
        def init(self, *args, **kwargs):
            return self
        
        def log(self, *args, **kwargs):
            pass
        
        def finish(self):
            pass
        
        def watch(self, *args, **kwargs):
            pass
        
        class config:
            @staticmethod
            def update(*args, **kwargs):
                pass
    
    mock = MockWandb()
    monkeypatch.setattr("wandb.init", mock.init)
    monkeypatch.setattr("wandb.log", mock.log)
    monkeypatch.setattr("wandb.finish", mock.finish)
    monkeypatch.setattr("wandb.watch", mock.watch)
    monkeypatch.setattr("wandb.config", mock.config)
    return mock


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    test_env_vars = {
        "CUDA_VISIBLE_DEVICES": "0",
        "WANDB_MODE": "offline",
        "PYTHONPATH": "/workspace"
    }
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)
    return test_env_vars


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    
    yield captured_output
    
    captured_output.close()


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "gpu: mark test to run only when GPU is available"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test that requires external data"
    )