"""
Simple Drift Detection Example

Basic demonstration of using the DriftPipeline API.
"""

from drift_autopsy import DriftPipeline, Dataset
from drift_autopsy.detectors import KSTest
import numpy as np


def main():
    print("Simple Drift Detection Example")
    print("=" * 50)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    
    # Reference data: normal distribution
    np.random.seed(42)
    ref_data = np.random.randn(1000, 5)
    
    # Test data: shifted distribution (drift!)
    test_data = np.random.randn(1000, 5) + 0.5  # Mean shift
    
    # Create datasets
    reference = Dataset.from_numpy(
        ref_data,
        feature_names=[f"feature_{i}" for i in range(5)]
    )
    
    test = Dataset.from_numpy(
        test_data,
        feature_names=[f"feature_{i}" for i in range(5)]
    )
    
    print(f"   Reference: {reference.shape}")
    print(f"   Test: {test.shape}")
    
    # Create and run pipeline
    print("\n2. Running drift detection...")
    
    pipeline = DriftPipeline(
        detector=KSTest(threshold=0.05),
        localizer="univariate",
        enable_localization=True,
    )
    
    result = pipeline.run(reference, test)
    
    # Print results
    print("\n3. Results:")
    print(f"   Drift Detected: {result.detection.drift_detected}")
    print(f"   Severity: {result.detection.severity.value}")
    print(f"   P-value: {result.detection.p_value:.6f}")
    
    if result.localization:
        print(f"\n   Drifted Features: {len(result.localization.drifted_features)}")
        for i, feature in enumerate(result.localization.drifted_features[:3], 1):
            score = result.localization.drift_scores[feature]
            print(f"     {i}. {feature}: score={score:.4f}")
    
    print(f"\n   Execution Time: {result.execution_time_seconds:.3f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
