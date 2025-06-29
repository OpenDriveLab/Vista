import sys
from pathlib import Path

import pytest
import torch


def test_python_version():
    """Test that Python version meets requirements."""
    assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"


def test_project_structure():
    """Test that the project structure is set up correctly."""
    project_root = Path(__file__).parent.parent
    
    # Check main directories exist
    assert (project_root / "vwm").exists(), "vwm package directory not found"
    assert (project_root / "tests").exists(), "tests directory not found"
    assert (project_root / "configs").exists(), "configs directory not found"
    
    # Check test subdirectories
    assert (project_root / "tests" / "unit").exists(), "unit tests directory not found"
    assert (project_root / "tests" / "integration").exists(), "integration tests directory not found"
    
    # Check important files
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml not found"
    assert (project_root / "README.md").exists(), "README.md not found"


def test_pytorch_available():
    """Test that PyTorch is available and working."""
    assert torch.__version__ is not None
    
    # Test basic tensor operations
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y
    assert torch.allclose(z, torch.tensor([5.0, 7.0, 9.0]))


def test_conftest_fixtures(temp_dir, mock_config, sample_tensor):
    """Test that conftest fixtures are available."""
    # Test temp_dir fixture
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Test mock_config fixture
    assert mock_config is not None
    assert "model" in mock_config
    assert mock_config.model.type == "test_model"
    
    # Test sample_tensor fixture
    assert sample_tensor.shape == (2, 3, 64, 64)
    assert sample_tensor.dtype == torch.float32


@pytest.mark.unit
def test_unit_marker():
    """Test that the unit marker works."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that the integration marker works."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that the slow marker works."""
    import time
    time.sleep(0.1)  # Simulate slow test
    assert True


def test_gpu_availability():
    """Test GPU availability (informational, not failing)."""
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        assert torch.cuda.device_count() > 0
    else:
        print("GPU is not available, tests will run on CPU")
        assert True  # Don't fail if GPU is not available


def test_import_main_package():
    """Test that the main package can be imported."""
    import vwm
    assert vwm is not None
    
    # Test submodules can be imported
    from vwm import models
    from vwm import data
    from vwm import modules
    
    assert models is not None
    assert data is not None
    assert modules is not None


class TestCoverageValidation:
    """Test class to validate coverage reporting works."""
    
    def test_covered_function(self):
        """This function should be covered."""
        result = self._helper_function(5)
        assert result == 10
    
    def _helper_function(self, x):
        """Helper function for coverage testing."""
        if x > 0:
            return x * 2
        else:
            return 0  # This line won't be covered in tests
    
    def test_partial_coverage(self):
        """Test to demonstrate partial coverage."""
        assert self._helper_function(3) == 6