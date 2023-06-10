<img src="https://raw.githubusercontent.com/ADMAntwerp/CounterPlots/main/_static/counterplots_logo.svg"><br>

--------------------------------------

CounterPlots: Plotting tool for counterfactuals
=======================================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![example workflow](https://github.com/ADMAntwerp/CounterPlots/actions/workflows/deployment.yml/badge.svg)](https://github.com/ADMAntwerp/CounterPlots/actions)
[![Code Coverage](https://codecov.io/gh/rmazzine/counterplotcoverage/branch/main/graph/badge.svg?token=TQYJSGEMP1)](https://codecov.io/gh/rmazzine/counterplotcoverage)
[![Known Vulnerabilities](https://snyk.io/test/github/ADMAntwerp/CounterPlots/badge.svg)](https://snyk.io/test/github/ADMAntwerp/CounterPlots)

Counterplots is a Python package that allows you to plot counterfactuals with easy integration with any counterfactual generation algorithm.

## Plot examples

### Greedy Plot

The greedy plot shows the greediest (feature change with the highest impact towards the opposite class) path from the factual instance until it reaches the counterfactual.

<img src="https://raw.githubusercontent.com/ADMAntwerp/CounterPlots/main/_static/counterplots_example_greedy.png">

### CounterShapley Plot

This chart shows each counterfactual feature change contribution to the counterfactual prediction. It uses Shapley values to calculate the contribution of each feature change.

<img src="https://raw.githubusercontent.com/ADMAntwerp/CounterPlots/main/_static/counterplots_example_countershapley.png">

### Constellation Plot

This chart shows the prediction score change for all possible feature change combinations.

<img src="https://raw.githubusercontent.com/ADMAntwerp/CounterPlots/main/_static/counterplots_example_constellation.png">

## Requirements
CounterPlots requires Python 3.8 or higher.

## Installation
With pip:
```bash
pip install counterplots
```

## Usage
To use CounterPlots, you just need the machine learning model predictor, and the factual and counterfactual points.
The example below uses a simple mock model:
```python
from counterplots import CreatePlot
import numpy as np

# Simple mock model for the predict_proba function which returns a probability for each input instance
def mock_predict_proba(data):
    out = []
    for x in data:
        if list(x) == [0.0, 0.0, 0.0]:
            out.append(0.0)
        elif list(x) == [1.0, 0.0, 0.0]:
            out.append(0.44)
        elif list(x) == [0.0, 1.0, 0.0]:
            out.append(0.4)
        elif list(x) == [0.0, 0.0, 1.0]:
            out.append(0.2)
        elif list(x) == [1.0, 1.0, 0.0]:
            out.append(0.3)
        elif list(x) == [0.0, 1.0, 1.0]:
            out.append(0.25)
        elif list(x) == [1.0, 0.0, 1.0]:
            out.append(0.4)
        elif list(x) == [1.0, 1.0, 1.0]:
            out.append(1.0)
    return np.array(out)

# Factual Instance
factual = np.array([0, 0, 0])
# Counterfactual Instance
cf = np.array([1, 1, 1])

# Create the plot object
cf_plots = CreatePlot(
    factual,
    cf,
    mock_predict_proba)

# Create the greedy plot
cf_plots.greedy('greedy_plot.png')
# Create the countershapley plot
cf_plots.countershapley('countershapley_plot.png')
# Create the constellation plot
cf_plots.constellation('constellation_plot.png')

# Print the countershapley values
print(cf_plots.countershapley_values())
```

In case you want to add custom names to the features, use the optional argument `feature_names`:
```python
cf_plots = CreatePlot(
    factual,
    cf,
    mock_predict_proba,
    feature_names=['feature1', 'feature2', 'feature3'])
```

In case you want to add custom labels to the factual and counterfactual points, use the optional argument `class_names`:
```python
cf_plots = CreatePlot(
    factual,
    cf,
    mock_predict_proba,
    class_names=['Factual', 'Counterfactual'])
```

## Using with Scikit-Learn

CounterPlots can be used with any machine learning model that has a `predict_proba` function. For example, with Scikit-Learn:
<details>

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from counterplots import CreatePlot

iris = load_iris()

X = iris.data
y = [0 if l == 0 else 1 for l in iris.target] # Makes it a binary classification problem

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

preds = clf.predict(X)

# For the factual point, takes an instance with 0 classification
factual = X[np.argwhere(preds == 0)[0]][0]
# For the counterfactual point, takes an instance with 1 classification
cf = X[np.argwhere(preds == 1)[0]][0]

cf_plots = CreatePlot(
    factual,
    cf,
    clf.predict_proba,
    feature_names=iris.feature_names,
    class_names={0: 'Setosa', 1: 'Non-Setosa'}
)


# Create the greedy plot
cf_plots.greedy('iris_greedy_plot.png')
# Create the countershapley plot
cf_plots.countershapley('iris_countershapley_plot.png')
# Create the constellation plot
cf_plots.constellation('iris_constellation_plot.png')

# Print the countershapley values
print(cf_plots.countershapley_values())
```
</details>

## Citation
If you use CounterPlots in your research, please cite the following paper:
```bibtex
```