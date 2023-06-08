import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from counterplots.utils.plots import make_greedy_plot, make_countershapley_plot, make_constellation_plot


class TestMakeGreedyPlot(unittest.TestCase):
    @patch('counterplots.utils.plots.plt')
    @patch('counterplots.utils.plots.mpl')
    def test_make_greedy_plot(self, mock_mpl, mock_plt):
        factual_score = 0.7
        features_data = [
            {'score': 0.4, 'name': 'Feature 1', 'factual': 'A', 'counterfactual': 'B'},
            {'score': 0.6, 'name': 'Feature 2', 'factual': 'X', 'counterfactual': 'Y'}
        ]
        class_names = {0: 'Class A', 1: 'Class B'}
        save_path = 'plot.png'

        make_greedy_plot(factual_score, features_data, class_names, save_path)


class TestMakeCounterShapleyPlot(unittest.TestCase):

    @patch('counterplots.utils.plots.plt')
    @patch('counterplots.utils.plots.mpl')
    def test_make_countershapley_plot(self, mock_mpl, mock_plt):
        factual_score = 0.0
        features_data = [{'x': 62.0,
                          'score': 0.4,
                          'name': 'Feature 1',
                          'factual': -4.0,
                          'counterfactual': -1.0},
                         {'x': 38.0,
                          'score': 0.6,
                          'name': 'Feature 2',
                          'factual': 1.0,
                          'counterfactual': -1.0}]
        class_names = {0: 'Class A', 1: 'Class B'}
        save_path = 'plot.png'

        make_countershapley_plot(factual_score, features_data, class_names, save_path)


class TestMakeConstellationPlot(unittest.TestCase):

    @patch('counterplots.utils.plots.plt')
    @patch('counterplots.utils.plots.mpl')
    def test_make_constellation_plot(self, mock_mpl, mock_plt):
        factual_score = 0.0
        single_points_chart = np.array([[0., 0.00519673],
                                        [1., 0.06478132],
                                        [2., 0.004789]])
        text_features = ['Feature 6 (-3.19627➜-0.3652)',
                         'Feature 8 (2.0093➜-3.15686)',
                         'Feature 9 (3.1568➜-2.41393)']
        mulitple_points_chart = np.array(
            [[np.array([0, 1]), 0.263131977249517],
             [np.array([0, 2]), 0.024619989978766642],
             [np.array([1, 2]), 0.24759475995893304]])
        mulitple_points_chart_y = [0.5, 1.0, 1.5]
        single_points = np.array([
            [6.00000000e+00, 5.19673099e-03],
            [8.00000000e+00, 6.47813170e-02],
            [9.00000000e+00, 4.78899690e-03]])
        class_names = {0: 'Class 1', 1: 'Class 0'}
        cf_score = 0.63
        point_to_pred = {0: 0.005196730993880675, 1: 0.06478131701518305, 2: 0.004788996904253137}
        save_path = 'plot.png'

        make_constellation_plot(factual_score, single_points_chart, text_features, mulitple_points_chart,
                                mulitple_points_chart_y, single_points, class_names, cf_score, point_to_pred,
                                save_path)

