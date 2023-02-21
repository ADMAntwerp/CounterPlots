from itertools import combinations
from typing import Callable, List, Dict

import numpy as np

from counterplots.utils.plots import make_greedy_plot, make_countershapley_plot, make_constellation_plot
from counterplots.utils.process import create_adapted_model_and_class, greedy_strategy, calc_shapley_values_between, \
    _convert_legendvalue
from counterplots.utils.verification import verify_factual_counterfactual_shape, verify_factual_cf,  \
    verify_feature_names, verify_model_prediction, verify_binary_class, verify_class_names


class CreatePlot:

    def __init__(
            self,
            factual: np.ndarray,
            cf: np.ndarray,
            model_pred: Callable,
            feature_names: List[str] = None,
            class_names: Dict[int, str] = None):

        # If feature_names is not provided, create it
        if feature_names is None:
            feature_names = [f'f{i+1}' for i in range(len(factual))]

        # If class_names is not provided, create it
        if class_names is None:
            class_names = {0: '0', 1: '1'}

        # Convert factual and cf to numpy arrays
        self.factual = np.array(factual)
        self.cf = np.array(cf)
        self.feature_names = feature_names
        self.class_names = class_names

        # Make verifications
        verify_factual_counterfactual_shape(factual, cf)
        verify_factual_cf(factual, cf)
        verify_feature_names(factual, feature_names)
        verify_model_prediction(model_pred, factual)
        verify_binary_class(model_pred, factual)
        verify_class_names(class_names)

        # Create features names if not provided
        if feature_names is None:
            self.feature_names = [f'Feat_idx_{i}' for i in range(len(factual))]

        # Create class names if not provided
        if class_names is None:
            self.class_names = {0: 'Class 0', 1: 'Class 1'}

        # Adapt model and class names
        self.adapted_model, self.class_names = create_adapted_model_and_class(
            factual=factual,
            model_pred=model_pred,
            class_names=class_names)

        self.factual_score = round(self.adapted_model(
            self.factual.reshape(1, -1))[0], 2)
        self.cf_score = round(self.adapted_model(self.cf.reshape(1, -1))[0], 2)

    def greedy(self, save_path: str = None):
        # Make a greedy search to find the best feature to change
        features_data = greedy_strategy(
            factual=self.factual,
            cf=self.cf,
            adapted_model=self.adapted_model,
            feature_names=self.feature_names)

        # Plot the greedy search
        make_greedy_plot(
            factual_score=self.factual_score,
            features_data=features_data,
            class_names=self.class_names,
            save_path=save_path)

    def countershapley(self, save_path: str = None):

        feat_contributions, feat_modified = calc_shapley_values_between(
            instance1=self.factual,
            instance2=self.cf,
            model=self.adapted_model,
            class_of_interest_index=0)

        # Create data for countershapley plot
        sum_contributions = np.sum(feat_contributions)
        current_score = self.factual_score
        features_data = []
        for feat, contrib in zip(feat_modified, feat_contributions):
            current_score += contrib

            factual_raw_value = self.factual[feat]
            cf_value = self.cf[feat]

            # Adjust text of modification values
            factual_value = _convert_legendvalue(factual_raw_value)
            counterfactual_value = _convert_legendvalue(cf_value)

            features_data.append({
                'x': contrib/sum_contributions*100,
                'score': round(current_score, 2),
                'name': self.feature_names[feat],

                'factual': factual_value,

                'counterfactual': counterfactual_value})

        # Plot the countershapley values
        make_countershapley_plot(
            self.factual_score, features_data, self.class_names, save_path)

    def countershapley_values(self):
        feat_contributions, feat_modified = calc_shapley_values_between(
            instance1=self.factual,
            instance2=self.cf,
            model=self.adapted_model,
            class_of_interest_index=0)

        return {
            'feature_names': [self.feature_names[feat] for feat in feat_modified],
            'feature_values': feat_contributions,
            'feature_indices': feat_modified
        }

    def constellation(self, save_path: str = None):

        # Get the index of the features that are different between the factual and counterfactual
        idx_feat_diff = np.where(self.factual != self.cf)[0]

        # Constellation need at least 2 features to be different
        if len(idx_feat_diff) < 2:
            print(
                'Showing greedy plot since constellation needs at least 2 features to be different')
            self.greedy(save_path)
            return

        text_features = []
        for i in idx_feat_diff:

            # Adjust text of modification values
            factual_raw_value = self.factual[i]
            cf_value = self.cf[i]

            factual_value = _convert_legendvalue(factual_raw_value)
            counterfactual_value = _convert_legendvalue(cf_value)

            factual_feat_text = str(factual_value)
            cf_feat_text = str(counterfactual_value)
            text_features.append(
                f'{self.feature_names[i]} ({factual_feat_text}➜{cf_feat_text})')

        # Create all possible combinations from 1 to the number of features that are different - 1
        idx_feat_comb = [list(combinations(idx_feat_diff, i))
                         for i in range(1, len(idx_feat_diff))]
        idx_feat_comb = [item for sublist in idx_feat_comb for item in sublist]

        # Dataset with modified points
        points_mod = np.array([self.factual] * len(idx_feat_comb))

        # Make the changes to the points
        points_mod = np.array(
            [np.array([points_mod[i][j] if j not in idx_feat_comb[i] else self.cf[j]
                       for j in range(len(self.factual))]) for i in range(len(idx_feat_comb))])

        # Get points predictions
        points_pred = self.adapted_model(points_mod)

        # Get single points with single change and their predictions
        single_points = np.array(
            [[comb[0], points_pred[idx]] for idx, comb in enumerate(idx_feat_comb) if len(comb) == 1])

        # Get multiple points with multiple changes and their predictions
        multiple_points = np.array(
            [[comb, points_pred[idx]] for idx, comb in enumerate(idx_feat_comb) if len(comb) > 1], dtype=object)

        # Points to chart points
        points_to_chart = {p: i for i, p in enumerate(single_points[:, 0])}
        point_to_pred = {i: p for i, p in enumerate(single_points[:, 1])}

        # Convert point features to chart points
        single_points_chart = np.array(
            [[points_to_chart[point_feat], point_pred] for point_feat, point_pred in single_points])
        mulitple_points_chart = np.array(
            [[np.array([points_to_chart[pf] for pf in point_feat]), point_pred] for point_feat, point_pred in
             multiple_points], dtype=object)

        # Multiple points y axis (0 index of mulitple_points_chart) must be the mean of the point values
        mulitple_points_chart_y = [np.mean(fp)
                                   for fp, _ in mulitple_points_chart]

        make_constellation_plot(
            factual_score=self.factual_score,
            single_points_chart=single_points_chart,
            text_features=text_features,
            mulitple_points_chart=mulitple_points_chart,
            mulitple_points_chart_y=mulitple_points_chart_y,
            single_points=single_points,
            class_names=self.class_names,
            cf_score=self.cf_score,
            point_to_pred=point_to_pred,
            save_path=save_path)
