"""
Tests for the visualization module.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.sparse as sp

from saplings.gasa.config import GASAConfig
from saplings.gasa.mask_builder import MaskFormat, MaskType
from saplings.gasa.visualization import MaskVisualizer


# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TestMaskVisualizer:
    """Tests for the MaskVisualizer class."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create the mask visualizer
        self.config = GASAConfig(
            visualize=True,
        )
        self.visualizer = MaskVisualizer(
            config=self.config,
        )
    
    def test_init(self):
        """Test initialization."""
        assert self.visualizer.config == self.config
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_visualize_mask_dense(self):
        """Test visualize_mask method with dense format."""
        # Create dense mask
        mask = np.ones((10, 10))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualize mask
            output_path = os.path.join(temp_dir, "mask.png")
            fig = self.visualizer.visualize_mask(
                mask=mask,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
                output_path=output_path,
                title="Test Mask",
                show=False,
            )
            
            # Check that the figure was created
            assert fig is not None
            
            # Check that the output file was created
            assert os.path.exists(output_path)
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_visualize_mask_sparse(self):
        """Test visualize_mask method with sparse format."""
        # Create sparse mask
        mask = sp.csr_matrix(np.ones((10, 10)))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualize mask
            output_path = os.path.join(temp_dir, "mask.png")
            fig = self.visualizer.visualize_mask(
                mask=mask,
                format=MaskFormat.SPARSE,
                mask_type=MaskType.ATTENTION,
                output_path=output_path,
                title="Test Mask",
                show=False,
            )
            
            # Check that the figure was created
            assert fig is not None
            
            # Check that the output file was created
            assert os.path.exists(output_path)
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_visualize_mask_block_sparse(self):
        """Test visualize_mask method with block-sparse format."""
        # Create block-sparse mask
        blocks = [
            {"row": 0, "col": 0, "size_row": 5, "size_col": 5, "block": [[1] * 5] * 5},
            {"row": 0, "col": 5, "size_row": 5, "size_col": 5, "block": [[1] * 5] * 5},
            {"row": 5, "col": 0, "size_row": 5, "size_col": 5, "block": [[1] * 5] * 5},
            {"row": 5, "col": 5, "size_row": 5, "size_col": 5, "block": [[1] * 5] * 5},
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualize mask
            output_path = os.path.join(temp_dir, "mask.png")
            fig = self.visualizer.visualize_mask(
                mask=blocks,
                format=MaskFormat.BLOCK_SPARSE,
                mask_type=MaskType.ATTENTION,
                output_path=output_path,
                title="Test Mask",
                show=False,
            )
            
            # Check that the figure was created
            assert fig is not None
            
            # Check that the output file was created
            assert os.path.exists(output_path)
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_visualize_mask_with_token_labels(self):
        """Test visualize_mask method with token labels."""
        # Create dense mask
        mask = np.ones((5, 5))
        
        # Create token labels
        token_labels = ["[CLS]", "This", "is", "a", "[SEP]"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualize mask
            output_path = os.path.join(temp_dir, "mask.png")
            fig = self.visualizer.visualize_mask(
                mask=mask,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
                output_path=output_path,
                title="Test Mask",
                show=False,
                token_labels=token_labels,
            )
            
            # Check that the figure was created
            assert fig is not None
            
            # Check that the output file was created
            assert os.path.exists(output_path)
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_visualize_mask_with_highlight_tokens(self):
        """Test visualize_mask method with highlight tokens."""
        # Create dense mask
        mask = np.ones((5, 5))
        
        # Create highlight tokens
        highlight_tokens = [0, 4]  # Highlight [CLS] and [SEP]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualize mask
            output_path = os.path.join(temp_dir, "mask.png")
            fig = self.visualizer.visualize_mask(
                mask=mask,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
                output_path=output_path,
                title="Test Mask",
                show=False,
                highlight_tokens=highlight_tokens,
            )
            
            # Check that the figure was created
            assert fig is not None
            
            # Check that the output file was created
            assert os.path.exists(output_path)
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_visualize_mask_sparsity(self):
        """Test visualize_mask_sparsity method."""
        # Create dense mask with 50% sparsity
        mask = np.zeros((10, 10))
        mask[:5, :5] = 1  # Upper-left quadrant is 1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualize mask sparsity
            output_path = os.path.join(temp_dir, "sparsity.png")
            fig = self.visualizer.visualize_mask_sparsity(
                mask=mask,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
                output_path=output_path,
                title="Test Sparsity",
                show=False,
            )
            
            # Check that the figure was created
            assert fig is not None
            
            # Check that the output file was created
            assert os.path.exists(output_path)
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
    def test_visualize_mask_comparison(self):
        """Test visualize_mask_comparison method."""
        # Create two masks
        mask1 = np.ones((10, 10))
        mask2 = np.zeros((10, 10))
        mask2[:5, :5] = 1  # Upper-left quadrant is 1
        
        # Create mask tuples
        masks = [
            (mask1, MaskFormat.DENSE, MaskType.ATTENTION, "Dense Mask"),
            (mask2, MaskFormat.DENSE, MaskType.ATTENTION, "Sparse Mask"),
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualize mask comparison
            output_path = os.path.join(temp_dir, "comparison.png")
            fig = self.visualizer.visualize_mask_comparison(
                masks=masks,
                output_path=output_path,
                title="Mask Comparison",
                show=False,
            )
            
            # Check that the figure was created
            assert fig is not None
            
            # Check that the output file was created
            assert os.path.exists(output_path)
    
    def test_convert_to_dense_dense(self):
        """Test _convert_to_dense method with dense format."""
        # Create dense mask
        mask = np.ones((10, 10))
        
        # Convert to dense
        dense_mask = self.visualizer._convert_to_dense(
            mask=mask,
            format=MaskFormat.DENSE,
        )
        
        # Check dense mask
        assert isinstance(dense_mask, np.ndarray)
        assert dense_mask.shape == (10, 10)
        assert np.all(dense_mask == 1)
    
    def test_convert_to_dense_sparse(self):
        """Test _convert_to_dense method with sparse format."""
        # Create sparse mask
        mask = sp.csr_matrix(np.ones((10, 10)))
        
        # Convert to dense
        dense_mask = self.visualizer._convert_to_dense(
            mask=mask,
            format=MaskFormat.SPARSE,
        )
        
        # Check dense mask
        assert isinstance(dense_mask, np.ndarray)
        assert dense_mask.shape == (10, 10)
        assert np.all(dense_mask == 1)
    
    def test_convert_to_dense_block_sparse(self):
        """Test _convert_to_dense method with block-sparse format."""
        # Create block-sparse mask
        blocks = [
            {"row": 0, "col": 0, "size_row": 5, "size_col": 5, "block": [[1] * 5] * 5},
            {"row": 0, "col": 5, "size_row": 5, "size_col": 5, "block": [[1] * 5] * 5},
            {"row": 5, "col": 0, "size_row": 5, "size_col": 5, "block": [[1] * 5] * 5},
            {"row": 5, "col": 5, "size_row": 5, "size_col": 5, "block": [[1] * 5] * 5},
        ]
        
        # Convert to dense
        dense_mask = self.visualizer._convert_to_dense(
            mask=blocks,
            format=MaskFormat.BLOCK_SPARSE,
        )
        
        # Check dense mask
        assert isinstance(dense_mask, np.ndarray)
        assert dense_mask.shape == (10, 10)
        assert np.all(dense_mask == 1)
    
    def test_calculate_mask_statistics(self):
        """Test calculate_mask_statistics method."""
        # Create dense mask with 50% sparsity
        mask = np.zeros((10, 10))
        mask[:5, :5] = 1  # Upper-left quadrant is 1
        
        # Calculate statistics
        stats = self.visualizer.calculate_mask_statistics(
            mask=mask,
            format=MaskFormat.DENSE,
        )
        
        # Check statistics
        assert stats["total_elements"] == 100
        assert stats["nonzero_elements"] == 25
        assert stats["zero_elements"] == 75
        assert stats["sparsity"] == 0.75
        assert stats["shape"] == (10, 10)
        
        # Check row and column statistics
        assert len(stats["row_nonzeros"]) == 10
        assert len(stats["col_nonzeros"]) == 10
        assert len(stats["row_sparsity"]) == 10
        assert len(stats["col_sparsity"]) == 10
        
        # Check summary statistics
        assert stats["min_row_nonzeros"] == 0
        assert stats["max_row_nonzeros"] == 5
        assert stats["avg_row_nonzeros"] == 2.5
        assert stats["min_col_nonzeros"] == 0
        assert stats["max_col_nonzeros"] == 5
        assert stats["avg_col_nonzeros"] == 2.5
    
    def test_save_statistics(self):
        """Test save_statistics method."""
        # Create statistics
        stats = {
            "total_elements": 100,
            "nonzero_elements": 25,
            "zero_elements": 75,
            "sparsity": 0.75,
            "shape": (10, 10),
            "row_nonzeros": np.array([5, 5, 5, 5, 5, 0, 0, 0, 0, 0]),
            "col_nonzeros": np.array([5, 5, 5, 5, 5, 0, 0, 0, 0, 0]),
            "row_sparsity": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "col_sparsity": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "min_row_nonzeros": 0,
            "max_row_nonzeros": 5,
            "avg_row_nonzeros": 2.5,
            "min_col_nonzeros": 0,
            "max_col_nonzeros": 5,
            "avg_col_nonzeros": 2.5,
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save statistics
            output_path = os.path.join(temp_dir, "stats.json")
            self.visualizer.save_statistics(
                statistics=stats,
                output_path=output_path,
            )
            
            # Check that the output file was created
            assert os.path.exists(output_path)
