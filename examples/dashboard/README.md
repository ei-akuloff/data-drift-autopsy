# Drift Autopsy Dashboard

Interactive web-based dashboard for visualizing drift detection results.

## Features

- **📈 Real-time Metrics**: Summary statistics and KPIs at a glance
- **📊 Time-Series Analysis**: Track drift scores across years
- **🔍 Detector Comparison**: Compare performance of different detectors
- **🎯 Feature-Level Insights**: Identify which features drifted
- **📉 Performance Monitoring**: Model accuracy degradation tracking
- **🎨 Interactive Visualizations**: Powered by Plotly for rich interactions

## Installation

Install dashboard dependencies:

```bash
pip install -e ".[dashboard]"
```

Or install individually:

```bash
pip install streamlit>=1.28.0 plotly>=5.17.0
```

## Quick Start

### 1. Generate Results

First, run the drift analysis demo to generate results:

```bash
python examples/quickstart/folktables_demo.py
```

This creates `outputs/folktables_drift_results.json`.

### 2. Launch Dashboard

```bash
streamlit run examples/dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Usage

### Sidebar Controls

- **Results File Path**: Path to JSON results file (default: `outputs/folktables_drift_results.json`)
- **Years Filter**: Select which years to display
- **Detectors Filter**: Choose which detectors to show
- **Display Options**: Toggle raw data tables

### Dashboard Sections

#### Summary Metrics
- Total years analyzed
- Drift events detected
- Average model accuracy
- Number of drifted features

#### Drift Analysis
- **Drift Score Timeline**: Line chart showing how drift scores evolved
- **Detector Comparison**: Bar chart comparing detectors across years
- **Performance Chart**: Model accuracy over time with delta
- **Severity Distribution**: Pie chart of drift severity levels

#### Feature-Level Analysis
- **Feature Drift Heatmap**: Color-coded matrix of feature drift over time
- **Top Drifted Features**: Bar chart of most drifted features

#### Detection Timeline
- Visual timeline showing when each detector flagged drift

## Customization

### Using Custom Results

Point to your own results file:

```python
# In sidebar, change "Results File Path" to:
my_results/custom_analysis.json
```

### Expected JSON Format

The dashboard expects JSON with this structure:

```json
{
  "yearly_results": {
    "2015": {
      "detectors": {
        "ks_test": {
          "drift_detected": true,
          "severity": "critical",
          "score": 0.0077,
          "p_value": 0.0000
        }
      },
      "localization": {
        "feature_drifts": [
          {
            "feature_name": "MAR",
            "drift_detected": true,
            "score": 0.123,
            "severity": "high"
          }
        ]
      },
      "metadata": {
        "test_accuracy": 0.7925,
        "accuracy_delta": -0.0059
      }
    }
  }
}
```

## Development

### Project Structure

```
examples/dashboard/
├── __init__.py           # Package init
├── app.py                # Main Streamlit application
├── data_loader.py        # JSON results parser
├── visualizations.py     # Plotly chart components
└── README.md             # This file
```

### Adding New Visualizations

1. Add chart function to `visualizations.py`:

```python
def create_my_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="year", y="metric")
    return fig
```

2. Use in `app.py`:

```python
from examples.dashboard import visualizations as viz

fig = viz.create_my_chart(data)
st.plotly_chart(fig)
```

## Tips

- **Performance**: Dashboard caches data loading for faster interactions
- **Responsiveness**: Uses wide layout for better visualization space
- **Export**: Click camera icon on charts to save as PNG
- **Zoom**: Click and drag on charts to zoom into specific regions
- **Compare**: Use legend to toggle detector visibility

## Troubleshooting

### File Not Found Error

```
❌ File not found: outputs/folktables_drift_results.json
```

**Solution**: Run the demo first:
```bash
python examples/quickstart/folktables_demo.py
```

### Import Error

```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**: Install dashboard dependencies:
```bash
pip install -e ".[dashboard]"
```

### Port Already in Use

```
Address already in use
```

**Solution**: Use a different port:
```bash
streamlit run examples/dashboard/app.py --server.port 8502
```

## Examples

### Filter by Year

In sidebar:
- Select only `2016, 2017, 2018` to focus on later years
- Charts update automatically

### Compare Specific Detectors

In sidebar:
- Deselect `PSI` and `MMD`
- Keep only `KS Test` and `CBPE`
- See focused comparison

### Export Data

1. Enable "Show Raw Data Tables" in sidebar
2. Scroll to "Raw Data Tables" section
3. Click download button on any table

## Links

- [Main Documentation](../../README.md)
- [Folktables Demo](../quickstart/folktables_demo.py)
- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Docs](https://plotly.com/python/)
