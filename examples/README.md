# Examples

This directory contains examples demonstrating the Drift Autopsy framework.

## Quick Start

### Simple Example
Basic drift detection on synthetic data:

```bash
cd examples/quickstart
python simple_example.py
```

### Folktables Demo
Temporal drift analysis on ACS Income data (CA 2014-2018):

```bash
cd examples/quickstart
python folktables_demo.py
```

This will:
1. Load ACS Income data for California (2014-2018)
2. Train a LogisticRegression model on 2014 data
3. Run multiple drift detectors (KS Test, PSI, MMD, CBPE) on each year
4. Save results to `outputs/folktables_drift_results.json`

**Requirements:**
```bash
pip install -e ".[demo]"
```

## Configuration-Driven Usage

Load a pipeline from YAML config:

```python
from drift_autopsy.config import ConfigLoader
from drift_autopsy import DriftPipeline, Dataset

# Load config
config = ConfigLoader.from_yaml("configs/examples/basic_drift_detection.yaml")

# Build pipeline (you'd need to implement this helper or construct manually)
pipeline = DriftPipeline(
    detector=config.detector.type,
    localizer=config.localizer.type if config.localizer else None,
    enable_localization=config.enable_localization,
    enable_rca=config.enable_rca,
)

# Run
result = pipeline.run(reference, test)
```

## What Each Detector Tests

- **KS Test**: Univariate distribution shifts (per feature)
- **PSI**: Population Stability Index - binned distribution comparison
- **MMD**: Multivariate distribution distance using kernel methods
- **CBPE**: Confidence-Based Performance Estimation - monitors model confidence shifts

## Expected Output

The Folktables demo creates:
- `outputs/folktables_drift_results.json`: Complete results for all years and detectors
- Console output with drift detection summary and top drifted features

## Next Steps

1. Modify detector thresholds in the scripts or configs
2. Try different feature selections
3. Add RCA (SHAP) analysis by setting `enable_rca=True`
4. Test on your own datasets using `Dataset.from_pandas()` or `Dataset.from_numpy()`
