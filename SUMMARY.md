# CounterPlots — Repository Summary

## Overview

**CounterPlots** is a Python package for visualizing **counterfactual explanations** in machine learning. It provides tools to understand how and why a model's prediction changes when features are modified from a factual instance to a counterfactual instance.

The package integrates with any ML model that exposes a prediction function (e.g., `predict_proba` in scikit-learn) and supports both binary classification outputs (single score) and two-column probability outputs.

## Package Structure

```
counterplots/
├── __init__.py                         # CreatePlot class — main entry point
└── utils/
    ├── __init__.py
    ├── _COLORS.py                      # Color palette constants for plots
    ├── approx.py                       # Shapley value approximation via subsampling
    ├── constellation_plot_dash.py      # Interactive Dash-based constellation plot
    ├── plots.py                        # Matplotlib plotting functions
    ├── process.py                      # Core computation (Shapley values, greedy strategy, model adaptation)
    └── verification.py                 # Input validation utilities
```

## Core Functionality

### `CreatePlot` (entry point)

The main class that orchestrates all plot generation. It accepts:
- `factual` (np.ndarray): The original instance.
- `cf` (np.ndarray): The counterfactual instance.
- `model_pred` (callable): Prediction function accepting batched input.
- `feature_names` (optional): Human-readable feature labels.
- `class_names` (optional): Dictionary mapping class indices to names.
- `decision_threshold` (float): Classification threshold (default 0.5).
- `sample_kwargs` (optional): Configuration for Shapley value subsampling.

### Plot Types

| Plot | Method | Description |
|------|--------|-------------|
| **Greedy** | `greedy()` | Shows the highest-impact feature change path from factual to counterfactual |
| **CounterShapley** | `countershapley()` | Waterfall-style chart of each feature change's Shapley contribution |
| **Constellation** | `constellation()` | Scatter plot of prediction scores for all feature change combinations |
| **Interactive Constellation** | `constellation_interactive()` | Dash-based interactive version with click-to-highlight |

### Additional Methods

- `countershapley_values()`: Returns raw CounterShapley values without plotting.

## Key Modules

### `utils/process.py`
- **`AdaptedModel`**: Wraps the user's model to normalize output to a single score in [0, 1] where lower values represent the factual class.
- **`create_adapted_model_and_class()`**: Constructs the adapted model and adjusts class names based on factual prediction.
- **`greedy_strategy()`**: Iteratively selects the feature change that most increases the prediction score.
- **`calc_shapley_values_between()`**: Computes Shapley values for each feature change between two instances.
- **`calc_contribution_graph()`**: Builds a dictionary mapping every subcoalition of feature changes to its prediction score. Supports serial and parallel execution.
- **`get_shapley_feature_contribution()`**: Calculates a single feature's Shapley value from the contribution graph.

### `utils/approx.py`
- Implements edge-based sampling of coalition space for approximating Shapley values in high-dimensional settings.
- **`sample_random_edges()`**: Samples Shapley comparison edges evenly distributed across features.
- **`_get_subcoalitions_sample()`**: Returns a deduplicated set of subcoalitions needed for sampled Shapley computation.
- Supports `ensure_outer_edges` to guarantee that empty→singleton and (N−1)→full comparisons are included.

### `utils/verification.py`
- Input validators: shape checks, factual/cf compatibility, feature name length, model callability, binary classification check, class name format.

### `utils/plots.py`
- `make_greedy_plot()`: Matplotlib scatter/line plot for greedy path.
- `make_countershapley_plot()`: Waterfall bar chart with arrow-shaped bars.
- `make_constellation_plot()`: Multi-point scatter plot with connection lines. Supports `highlight_changed_feature` to emphasize a single feature's coalitions.
- `make_constellation_plot_dash()`: Launches interactive Dash app via `ConstellationPlotDash`.

### `utils/constellation_plot_dash.py`
- `ConstellationPlotDash`: Full Dash application with click-based coalition highlighting, reset button, and hover tooltips.

## Current State of Tests

### Existing Test Files

| File | Status | Notes |
|------|--------|-------|
| `tests/test____init__.py` | 8/10 pass | 2 tests fail due to stale mocks (mock returns 2-tuple but code expects 3-tuple) |
| `tests/utils/test_verification.py` | All pass | 7 tests covering all verification functions |
| `tests/utils/test_process.py` | 7/9 pass | `test_calc_adapted_instance` fails (array comparison), `test_calc_contribution_graph` fails (API signature changed) |
| `tests/utils/test_approx.py` | Import error | Tests reference `get_subcoalitions_sample` but function is `_get_subcoalitions_sample`; also references `return_as_edges` parameter that doesn't exist |
| `tests/utils/test_plots.py` | All pass | 3 tests covering plot generation |

### Known Issues

1. **`test_approx.py` import error**: The tests import `get_subcoalitions_sample` (public), but the actual function name is `_get_subcoalitions_sample` (private). Tests also use a `return_as_edges` parameter that was not implemented.
2. **`test_calc_contribution_graph` API mismatch**: The test passes `indices_feature_diff` and `class_of_interest_index` as positional args, but `calc_contribution_graph` now derives `indices_feature_diff` internally from the instances.
3. **`test_countershapley` / `test_countershapley_values` mock mismatch**: Mocks return a 2-tuple, but `calc_shapley_values_between` now returns a 3-tuple (contributions, indices, contribution_graph).
4. **`test_calc_adapted_instance`**: Uses `assertEqual` on numpy arrays, which raises an ambiguity error. Should use `np.testing.assert_array_equal`.

## Dependencies

**Core** (from `requirements.txt`):
- `numpy >= 1.24.3`
- `matplotlib >= 3.7.1`

**Extended** (from `pyproject.toml`):
- `dash >= 4.0.0`, `plotly >= 6.5.2` (interactive constellation plot)
- `scikit-learn`, `shap`, `lime`, `dice-ml` (experimental notebooks)

## CI/CD

- **GitHub Actions** workflow at `.github/workflows/deployment.yml`
- Linting with `flake8`, testing with `pytest --cov`, coverage upload to Codecov
- Package build via `setup.py sdist`, deployment via `twine upload`
