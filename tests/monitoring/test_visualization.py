"""
Tests for the monitoring visualization components.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from saplings.gasa.config import GASAConfig
from saplings.gasa.mask_builder import MaskType
from saplings.monitoring.config import MonitoringConfig, VisualizationFormat
from saplings.monitoring.visualization import GASAHeatmap, MaskFormat, PerformanceVisualizer

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Check if plotly is available
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@pytest.fixture
def gasa_heatmap():
    """Create a GASAHeatmap instance for testing."""
    config = MonitoringConfig(
        visualization_output_dir=tempfile.mkdtemp(),
        visualization_format=VisualizationFormat.PNG,
    )
    gasa_config = GASAConfig(
        max_hops=2,
        mask_strategy="binary",
    )
    return GASAHeatmap(config=config, gasa_config=gasa_config)


@pytest.fixture
def performance_visualizer():
    """Create a PerformanceVisualizer instance for testing."""
    config = MonitoringConfig(
        visualization_output_dir=tempfile.mkdtemp(),
        visualization_format=VisualizationFormat.PNG,
    )
    return PerformanceVisualizer(config=config)


@pytest.fixture
def sample_mask():
    """Create a sample attention mask for testing."""
    # Create a 10x10 mask with some patterns
    mask = np.zeros((10, 10))

    # Add diagonal (self-attention)
    np.fill_diagonal(mask, 1.0)

    # Add some connections
    mask[0, 1:5] = 1.0
    mask[1, 0] = 1.0
    mask[2, 3:7] = 1.0
    mask[5, 6:9] = 1.0

    return mask


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
def test_gasa_heatmap_static_visualization(gasa_heatmap, sample_mask):
    """Test static visualization of GASA heatmap."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        # Visualize mask
        fig = gasa_heatmap._visualize_mask_static(
            mask=sample_mask,
            format=MaskFormat.DENSE,
            mask_type=MaskType.ATTENTION,
            output_path=tmp.name,
            title="Test Mask",
            show=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
def test_gasa_heatmap_interactive_visualization(gasa_heatmap, sample_mask):
    """Test interactive visualization of GASA heatmap."""
    with tempfile.NamedTemporaryFile(suffix=".html") as tmp:
        # Visualize mask
        fig = gasa_heatmap._visualize_mask_interactive(
            mask=sample_mask,
            format=MaskFormat.DENSE,
            mask_type=MaskType.ATTENTION,
            output_path=tmp.name,
            title="Test Mask",
            show=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
def test_gasa_heatmap_sparsity_visualization(gasa_heatmap, sample_mask):
    """Test sparsity visualization of GASA heatmap."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        # Visualize mask sparsity
        fig = gasa_heatmap.visualize_mask_sparsity(
            mask=sample_mask,
            format=MaskFormat.DENSE,
            mask_type=MaskType.ATTENTION,
            output_path=tmp.name,
            title="Test Mask Sparsity",
            show=False,
            interactive=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
def test_gasa_heatmap_comparison_visualization(gasa_heatmap, sample_mask):
    """Test comparison visualization of GASA heatmap."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        # Create a second mask
        mask2 = np.zeros((10, 10))
        np.fill_diagonal(mask2, 1.0)
        mask2[0, 1:3] = 1.0
        mask2[3, 4:7] = 1.0

        # Visualize mask comparison
        fig = gasa_heatmap._visualize_mask_comparison_static(
            masks=[
                (sample_mask, MaskFormat.DENSE, MaskType.ATTENTION, "Mask 1"),
                (mask2, MaskFormat.DENSE, MaskType.ATTENTION, "Mask 2"),
            ],
            output_path=tmp.name,
            title="Test Mask Comparison",
            show=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
def test_performance_visualizer_latency(performance_visualizer):
    """Test latency visualization."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        # Create sample latency data
        latencies = {
            "Component A": [10.5, 12.3, 9.8, 11.2, 10.9],
            "Component B": [15.2, 14.8, 16.1, 15.5, 14.9],
            "Component C": [5.3, 6.1, 5.8, 5.5, 6.0],
        }

        # Visualize latency
        fig = performance_visualizer.visualize_latency(
            latencies=latencies,
            output_path=tmp.name,
            title="Test Latency",
            show=False,
            interactive=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
def test_performance_visualizer_throughput(performance_visualizer):
    """Test throughput visualization."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        # Create sample throughput data
        throughputs = {
            "Component A": [100, 120, 110, 115, 105],
            "Component B": [80, 85, 82, 88, 90],
            "Component C": [150, 155, 160, 145, 152],
        }

        # Visualize throughput
        fig = performance_visualizer.visualize_throughput(
            throughputs=throughputs,
            output_path=tmp.name,
            title="Test Throughput",
            show=False,
            interactive=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
def test_performance_visualizer_error_rate(performance_visualizer):
    """Test error rate visualization."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        # Create sample error rate data
        error_rates = {
            "Component A": [0.05, 0.03, 0.04, 0.02, 0.03],
            "Component B": [0.10, 0.12, 0.09, 0.11, 0.10],
            "Component C": [0.01, 0.02, 0.01, 0.03, 0.02],
        }

        # Visualize error rate
        fig = performance_visualizer.visualize_error_rate(
            error_rates=error_rates,
            output_path=tmp.name,
            title="Test Error Rate",
            show=False,
            interactive=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
def test_performance_visualizer_interactive(performance_visualizer):
    """Test interactive performance visualization."""
    with tempfile.NamedTemporaryFile(suffix=".html") as tmp:
        # Create sample performance data
        latencies = {
            "Component A": [10.5, 12.3, 9.8, 11.2, 10.9],
            "Component B": [15.2, 14.8, 16.1, 15.5, 14.9],
            "Component C": [5.3, 6.1, 5.8, 5.5, 6.0],
        }

        # Visualize latency interactively
        fig = performance_visualizer.visualize_latency(
            latencies=latencies,
            output_path=tmp.name,
            title="Test Latency",
            show=False,
            interactive=True,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


def test_convert_to_dense(gasa_heatmap, sample_mask):
    """Test conversion to dense format."""
    # Test with already dense mask
    dense_mask = gasa_heatmap._convert_to_dense(sample_mask, MaskFormat.DENSE)
    assert np.array_equal(dense_mask, sample_mask)

    # Test with sparse tensor format
    sparse_mask = []
    for i in range(sample_mask.shape[0]):
        for j in range(sample_mask.shape[1]):
            if sample_mask[i, j] > 0:
                sparse_mask.append({"i": i, "j": j, "v": sample_mask[i, j]})

    dense_mask = gasa_heatmap._convert_to_dense(sparse_mask, MaskFormat.SPARSE_TENSOR)
    assert np.array_equal(dense_mask, sample_mask)


def test_invalid_mask_format(gasa_heatmap, sample_mask):
    """Test handling of invalid mask format."""
    with pytest.raises(ValueError):
        gasa_heatmap._convert_to_dense(sample_mask, "invalid_format")
