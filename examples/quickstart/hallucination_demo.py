"""
Hallucination Risk Detection Example

Demonstrates Module 1: Confidence vs Density detector.

A model prediction is "hallucination risk" when the model is
simultaneously high-confidence AND the input is far from the
training distribution — the "confident but likely wrong" regime.

Run:
    python examples/quickstart/hallucination_demo.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from drift_autopsy import HallucinationRiskDetector, Dataset


# ---------------------------------------------------------------------------
# 1. Simulate training data & fit a model
# ---------------------------------------------------------------------------

def make_data(rng: np.random.Generator):
    """
    Create a clean reference distribution and a test set that mixes:
      - in-distribution samples  (should be SAFE)
      - out-of-distribution samples with high model confidence
        (should be flagged as HALLUCINATION RISK)
    """
    # Reference: 400 samples, 6 features, centred at 0
    X_ref = rng.normal(loc=0.0, scale=1.0, size=(400, 6))
    y_ref = (X_ref[:, 0] + X_ref[:, 1] > 0).astype(int)  # simple linear boundary

    # Test — in-distribution portion (200 samples, same manifold)
    X_in  = rng.normal(loc=0.0, scale=1.0, size=(200, 6))

    # Test — out-of-distribution portion (100 samples, shifted far away)
    # The model will extrapolate here and produce high-confidence softmax outputs
    X_out = rng.normal(loc=6.0, scale=0.3, size=(100, 6))

    X_test = np.vstack([X_in, X_out])

    return X_ref, y_ref, X_test


def main():
    print("Hallucination Risk Detection Demo")
    print("=" * 55)

    rng = np.random.default_rng(0)
    X_ref, y_ref, X_test = make_data(rng)

    # -----------------------------------------------------------------------
    # 2. Train a logistic regression model on the reference data
    # -----------------------------------------------------------------------
    print("\n1. Training logistic regression on reference data...")
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_ref, y_ref)

    # Get prediction probabilities for both splits
    proba_ref  = clf.predict_proba(X_ref)   # (400, 2)
    proba_test = clf.predict_proba(X_test)  # (300, 2)

    print(f"   Reference : {X_ref.shape[0]} samples × {X_ref.shape[1]} features")
    print(f"   Test      : {X_test.shape[0]} samples "
          f"({200} in-distribution + {100} out-of-distribution)")

    # -----------------------------------------------------------------------
    # 3. Wrap in Dataset
    # -----------------------------------------------------------------------
    reference_dataset = Dataset(
        data=X_ref,
        feature_names=[f"feature_{i}" for i in range(X_ref.shape[1])],
        prediction_probabilities=proba_ref,
    )

    test_dataset = Dataset(
        data=X_test,
        feature_names=[f"feature_{i}" for i in range(X_test.shape[1])],
        prediction_probabilities=proba_test,
    )

    # -----------------------------------------------------------------------
    # 4. Run HallucinationRiskDetector — try all four density methods
    # -----------------------------------------------------------------------
    print("\n2. Running hallucination risk detection...\n")

    methods = ["mahalanobis", "knn", "isolation_forest", "kde"]
    results = {}

    for method in methods:
        detector = HallucinationRiskDetector(
            density_method=method,
            confidence_threshold=0.80,
            distance_threshold=0.50,
            random_state=0,
        )
        result = detector.fit_detect(reference_dataset, test_dataset)
        results[method] = result

        print(f"  [{method}]")
        print(f"    Hallucination rate : {result.hallucination_rate:.1%}")
        print(f"    Severity           : {result.severity.value.upper()}")
        print(f"    Samples flagged    : {result.n_hallucination_risk} / {len(X_test)}")
        print()

    # -----------------------------------------------------------------------
    # 5. Detailed view of the best method (Mahalanobis)
    # -----------------------------------------------------------------------
    print("3. Deep-dive: Mahalanobis method")
    print("-" * 40)

    result = results["mahalanobis"]

    # -- Core outputs --------------------------------------------------------
    print(f"\n  hallucination_rate  : {result.hallucination_rate:.3f}")
    print(f"  severity            : {result.severity}")

    # -- Four-quadrant breakdown --------------------------------------------
    q = result.quadrant_counts
    print(f"\n  Interpretability quadrants (n={len(X_test)})")
    print(f"    ✅ Safe (low dist, high conf)           : {q['safe']:>4}")
    print(f"    😐 Uncertain safe (low dist, low conf)  : {q['uncertain_safe']:>4}")
    print(f"    ⚠️  Honest UQ (high dist, low conf)     : {q['uncertain_honest']:>4}")
    print(f"    🚨 Hallucination risk (high dist+conf)  : {q['hallucination_risk']:>4}")

    # -- Flagged indices (first 10) -----------------------------------------
    print(f"\n  flagged_indices (first 10): {result.flagged_indices[:10].tolist()}")
    print(f"  (expected: mostly indices 200–299, the OOD block)")

    # -- Score distributions ------------------------------------------------
    in_scores  = result.hallucination_scores[:200]
    out_scores = result.hallucination_scores[200:]
    print(f"\n  Mean hallucination score — in-dist  : {in_scores.mean():.3f}")
    print(f"  Mean hallucination score — out-dist : {out_scores.mean():.3f}")
    print(f"  (out-of-distribution should score much higher)")

    # -----------------------------------------------------------------------
    # 6. to_dict() — ready for JSON export
    # -----------------------------------------------------------------------
    print("\n4. Serialising result to dict (for dashboard / logging)...")
    d = result.to_dict()
    print(f"   Keys: {list(d.keys())}")
    print(f"   hallucination_rate : {d['hallucination_rate']:.3f}")
    print(f"   severity           : {d['severity']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
