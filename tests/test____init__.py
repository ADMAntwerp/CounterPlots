import numpy as np
import unittest
from unittest.mock import patch
from counterplots import CreatePlot


class CreatePlotTests(unittest.TestCase):

    def setUp(self):
        # Create dummy data for testing
        self.factual = np.array([1, 2, 3])
        self.cf = np.array([4, 5, 6])
        self.model_pred = lambda x: np.array([np.array([0]) if sum(r) < 10 else np.array([1]) for r in x])
        self.feature_names = ['feat1', 'feat2', 'feat3']
        self.class_names = {0: '0', 1: '1'}

    def test_init(self):
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        self.assertTrue(np.array_equal(create_plot.factual, self.factual))
        self.assertTrue(np.array_equal(create_plot.cf, self.cf))
        self.assertEqual(create_plot.feature_names, self.feature_names)
        self.assertEqual(create_plot.class_names, self.class_names)
        self.assertEqual(create_plot.factual_score, 0)
        self.assertEqual(create_plot.cf_score, 1)

    def test_init_no_feat_name(self):
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, None, self.class_names)
        self.assertTrue(np.array_equal(create_plot.factual, self.factual))
        self.assertTrue(np.array_equal(create_plot.cf, self.cf))
        self.assertEqual(create_plot.feature_names, ['f1', 'f2', 'f3'])
        self.assertEqual(create_plot.class_names, self.class_names)
        self.assertEqual(create_plot.factual_score, 0)
        self.assertEqual(create_plot.cf_score, 1)

    def test_init_no_class_name(self):
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, None)
        self.assertTrue(np.array_equal(create_plot.factual, self.factual))
        self.assertTrue(np.array_equal(create_plot.cf, self.cf))
        self.assertEqual(create_plot.feature_names, self.feature_names)
        self.assertEqual(create_plot.class_names, {0: '0', 1: '1'})
        self.assertEqual(create_plot.factual_score, 0)
        self.assertEqual(create_plot.cf_score, 1)

    @patch('counterplots.make_greedy_plot')
    @patch('counterplots.greedy_strategy')
    def test_greedy(self, mock_greedy_strategy, mock_make_greedy_plot):
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        create_plot.greedy(save_path=None)  # Just check if it runs without error

    @patch('counterplots.make_countershapley_plot')
    @patch('counterplots._convert_legendvalue')
    @patch('counterplots.calc_shapley_values_between')
    def test_countershapley(self, mock_calc_shapley_values_between, mock__convert_legendvalue,
                            mock_make_countershapley_plot):

        mock_calc_shapley_values_between.return_value = ((0.4, 0.2, 0.1), (0, 1, 2))

        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        create_plot.countershapley(save_path=None)  # Just check if it runs without error

    @patch('counterplots.calc_shapley_values_between')
    def test_countershapley_values(self, mock_calc_shapley_values_between):
        mock_calc_shapley_values_between.return_value = ((0.4, 0.2, 0.1), (0, 1, 2))
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        values = create_plot.countershapley_values()
        expected_values = {
            'feature_names': self.feature_names,
            'feature_values': (0.4, 0.2, 0.1),
            'feature_indices': (0, 1, 2)
        }
        self.assertEqual(values, expected_values)

    @patch('counterplots.make_constellation_plot')
    def test_constellation(self, mock_make_constellation_plot):
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        create_plot.constellation(save_path=None)  # Just check if it runs without error

    @patch('counterplots.make_constellation_plot')
    @patch('counterplots.CreatePlot.greedy')
    def test_constellation_few_changes(self, mock_greedy, mock_make_constellation_plot):
        create_plot = CreatePlot(np.array([1, 2, 3]), np.array([1, 2, 4]),
                                 self.model_pred, self.feature_names, self.class_names)
        create_plot.constellation(save_path=None)

        # Check if greedy is called
        mock_greedy.assert_called_once_with(None)

        # Check if constellation is not called
        mock_make_constellation_plot.assert_not_called()
