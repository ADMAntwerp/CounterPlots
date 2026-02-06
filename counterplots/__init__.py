from itertools import combinations
from typing import Callable, List, Dict
import logging
import numpy as np

from counterplots.utils.plots import make_greedy_plot, make_countershapley_plot, make_constellation_plot, make_constellation_plot_dash
from counterplots.utils.process import create_adapted_model_and_class, greedy_strategy, calc_shapley_values_between, \
    _convert_legendvalue
from counterplots.utils.verification import verify_factual_counterfactual_shape, verify_factual_cf,  \
    verify_feature_names, verify_model_prediction, verify_binary_class, verify_class_names

logging.basicConfig()
logger = logging.getLogger(__name__)


class CreatePlot:

    def __init__(
            self,
            factual: np.ndarray,
            cf: np.ndarray,
            model_pred: Callable,
            feature_names: List[str] = None,
            class_names: Dict[int, str] = None,
            decision_threshold = 0.5
            sample_kwargs=None
            ):
        """

        Args:
            factual (np.ndarray): The factual instance
            cf (np.ndarray): The counterfactual instance
            model_pred (callable): A function that takes an instance and returns a model prediction score.
            feature_names (List[str]): Array of feature names.
            class_names (Dict[int, str]): Dictionary that maps class integers to their name. Used for plotting.
            sample_kwargs (Dict | None): Dictionary containing configuration for subsampling Shapley values.
                Must contain the keys "n" (sample size). 
                May specify the key "ensure_outer_edges" to ensure the outer edges of the contribution graph is sampled. 
                Outer edges compare contributions between: {} and {i}, S and S \ {i}.
                Default is True for stable solutions.
        """

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
        self.decision_threshold = decision_threshold
        self.sample_kwargs = sample_kwargs
        self.idx_feat_diff = np.where(self.factual != self.cf)[0]
        self.predict_fn, self.class_names = create_adapted_model_and_class(
            factual=factual,
            model_pred=model_pred,
            class_names=class_names
        )
        # Calculate all model scores for all subcoalitions or a sample thereof
        self.feat_contributions, self.adapted_feat_idxs, self.contribution_graph = calc_shapley_values_between(
            instance1 = self.factual,
            instance2 = self.cf,
            predict_fn=self.predict_fn,
            sample_kwargs=self.sample_kwargs,
            return_contribution_graph=True
        )
        
        self.factual_score = self.contribution_graph[()]
        self.cf_score = self.contribution_graph[tuple(self.idx_feat_diff)]

        # Make verifications
        verify_factual_counterfactual_shape(factual, cf)
        verify_factual_cf(factual, cf)
        verify_feature_names(factual, feature_names)
        verify_model_prediction(model_pred, factual)
        verify_binary_class(model_pred, factual)
        verify_class_names(class_names)

    def greedy(self, save_path: str = None):
        # Make a greedy search to find the best feature to change
        features_data = greedy_strategy(
            factual=self.factual,
            cf=self.cf,
            predict_fn=self.predict_fn,
            feature_names=self.feature_names
            )

        # Plot the greedy search
        fig, ax = make_greedy_plot(
            factual_score=self.factual_score,
            features_data=features_data,
            class_names=self.class_names,
            save_path=save_path,
            decision_threshold=self.decision_threshold
            )
        return fig, ax

    def countershapley(self, save_path: str = None):

        # Create data for countershapley plot
        sum_contributions = np.sum(self.feat_contributions)
        current_score = self.factual_score
        features_data = []
        for feat, contrib in zip(self.adapted_feat_idxs, self.feat_contributions):
            current_score += contrib

            factual_raw_value = self.factual[feat]
            cf_value = self.cf[feat]

            # Adjust text of modification values
            factual_value = _convert_legendvalue(factual_raw_value)
            counterfactual_value = _convert_legendvalue(cf_value)

            features_data.append({
                'x': contrib/sum_contributions*100,
                'score': current_score,
                'name': self.feature_names[feat],

                'factual': factual_value,

                'counterfactual': counterfactual_value})

        # Plot the countershapley values
        fig, ax = make_countershapley_plot(
            self.factual_score, 
            features_data, 
            self.class_names, 
            save_path,
            )

        return fig, ax

    def countershapley_values(self):
        return {
            'feature_names': [self.feature_names[feat] for feat in self.adapted_feat_idxs],
            'feature_values': self.feat_contributions,
            'feature_indices': self.adapted_feat_idxs
        }

    def constellation(self, save_path: str = None, highlight_changed_feature=None):
        """
        Create a constellation plot showing feature contributions to the counterfactual.
        
        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
            highlight_changed_feature (int, optional): Index of the changed feature to highlight 
                (0 to N-1, where N is the number of features that differ between factual and 
                counterfactual). When specified, only the coalitions containing this feature 
                change will be emphasized in the plot. Defaults to None (all features shown 
                with equal emphasis).
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        # Constellation need at least 2 features to be different
        if len(self.idx_feat_diff) < 2:
            print(
                'Showing greedy plot since constellation needs at least 2 features to be different')
            self.greedy(save_path)
            return

        text_features = []
        for i in self.idx_feat_diff:
            factual_raw_value = self.factual[i]
            cf_value = self.cf[i]

            factual_value = _convert_legendvalue(factual_raw_value)
            counterfactual_value = _convert_legendvalue(cf_value)

            factual_feat_text = str(factual_value)
            cf_feat_text = str(counterfactual_value)
            text_features.append(
                f'{self.feature_names[i]} ({factual_feat_text}➜{cf_feat_text})')

        # Get single points with single change and their predictions - These are plotted larger
        single_points = np.array([
            [comb[0], self.contribution_graph[comb]] 
            for comb in self.contribution_graph.keys()
            if len(comb) == 1
        ])

        # Get multiple points with multiple changes and their predictions - these are plotted smaller
        multiple_points = np.array([
            [comb, self.contribution_graph[comb]] 
            for comb in self.contribution_graph.keys()
            if len(comb) > 1
        ], dtype=object)

        # Points to chart points
        points_to_chart = {p: i for i, p in enumerate(single_points[:, 0])}
        point_to_pred = {i: p for i, p in enumerate(single_points[:, 1])}

        # Convert point features to chart points
        single_points_chart = np.array([
            [points_to_chart[point_feat], point_pred] 
            for point_feat, point_pred in single_points
            ])
        multiple_points_chart = np.array([
            [
                np.array([points_to_chart[pf] for pf in point_feat]), 
                point_pred
            ] for point_feat, point_pred in multiple_points
        ], dtype=object)

        # Multiple points y axis (0 index of multiple_points_chart) must be the mean of the point values
        multiple_points_chart_y = [np.mean(fp) for fp, _ in multiple_points_chart]

        fig, ax = make_constellation_plot(
            factual_score=self.factual_score,
            single_points_chart=single_points_chart,
            text_features=text_features,
            multiple_points_chart=multiple_points_chart,
            multiple_points_chart_y=multiple_points_chart_y,
            single_points=single_points,
            class_names=self.class_names,
            cf_score=self.cf_score,
            point_to_pred=point_to_pred,
            save_path=save_path,
            highlight_changed_feature=highlight_changed_feature,
            decision_threshold=self.decision_threshold
        )

        return fig, ax

    def constellation_interactive(self, save_path: str = None, port=8050, debug=False):
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
        idx_feat_comb = [list(combinations(idx_feat_diff, i)) for i in range(1, len(idx_feat_diff))]
        idx_feat_comb = [item for sublist in idx_feat_comb for item in sublist]

        # Dataset with modified points
        points_mod = np.array([self.factual] * len(idx_feat_comb))

        # Make the changes to the points
        points_mod = np.array([
            np.array([
                points_mod[i][j] if j not in idx_feat_comb[i] else self.cf[j]
                for j in range(len(self.factual))]) for i in range(len(idx_feat_comb)
            )]
        )

        # Get points predictions
        points_pred = self.predict_fn(points_mod)

        # Get single points with single change and their predictions - These are plotted larger
        single_points = np.array([
            [comb[0], points_pred[idx]] 
            for idx, comb in enumerate(idx_feat_comb) 
            if len(comb) == 1
        ])

        # Get multiple points with multiple changes and their predictions - these are plotted smaller
        multiple_points = np.array([
            [comb, points_pred[idx]] 
            for idx, comb in enumerate(idx_feat_comb) 
            if len(comb) > 1
        ], dtype=object)

        # Points to chart points
        points_to_chart = {p: i for i, p in enumerate(single_points[:, 0])}
        point_to_pred = {i: p for i, p in enumerate(single_points[:, 1])}

        # Convert point features to chart points
        single_points_chart = np.array([
            [points_to_chart[point_feat], point_pred] 
            for point_feat, point_pred in single_points
            ])
        multiple_points_chart = np.array([
            [
                np.array([points_to_chart[pf] for pf in point_feat]), 
                point_pred
            ] for point_feat, point_pred in multiple_points
        ], dtype=object)

        # Multiple points y axis (0 index of multiple_points_chart) must be the mean of the point values
        multiple_points_chart_y = [np.mean(fp) for fp, _ in multiple_points_chart]

        fig = make_constellation_plot_dash(
            factual_score=self.factual_score,
            single_points_chart=single_points_chart,
            text_features=text_features,
            multiple_points_chart=multiple_points_chart,
            multiple_points_chart_y=multiple_points_chart_y,
            single_points=single_points,
            class_names=self.class_names,
            cf_score=self.cf_score,
            point_to_pred=point_to_pred,
            decision_threshold=self.decision_threshold,
            port=8050,
            debug=debug
        )


        return fig

