"""
RAPIDS tests for accelerated-python tutorial.
These tests validate that cuDF and cuML are installed and functional.
These tests should be run in the RAPIDS venv.
"""

import pytest
import numpy as np


def test_cudf():
    """Test that cuDF works by performing DataFrame operations."""
    import cudf
    import pandas as pd

    # Create a cuDF DataFrame
    df = cudf.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50]
    })

    # Perform operations
    df['c'] = df['a'] + df['b']
    result = df['c'].sum()

    # Verify computation (1+10 + 2+20 + 3+30 + 4+40 + 5+50 = 165)
    assert result == 165

    # Test groupby operation
    df['group'] = [0, 0, 1, 1, 1]
    grouped = df.groupby('group')['a'].sum()
    assert len(grouped) == 2


def test_cuml():
    """Test that cuML works by training a simple model."""
    import cuml
    from cuml.cluster import KMeans
    import numpy as np

    # Create simple synthetic data
    n_samples = 100
    n_features = 2
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Train a simple KMeans model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Predict on the same data
    labels = kmeans.predict(X)

    # Verify we got labels and they're in expected range
    assert len(labels) == n_samples
    assert all(0 <= label < 3 for label in labels)
