import numpy as np
import unittest
from unittest.mock import patch
from counterplots import CreatePlot
import matplotlib.pyplot as plt


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

    def test_greedy(self):
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        fig, ax = create_plot.greedy(save_path=None)
        plt.close(fig)

    @patch('counterplots._convert_legendvalue')
    @patch('counterplots.calc_shapley_values_between')
    def test_countershapley(self, mock_calc_shapley_values_between, mock__convert_legendvalue):
        mock_contribution_graph = {(): 0.2, (0, 1, 2): 0.8}
        mock_calc_shapley_values_between.return_value = ((0.4, 0.2, 0.1), (0, 1, 2), mock_contribution_graph)

        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        fig, ax = create_plot.countershapley(save_path=None)
        plt.close(fig)

    @patch('counterplots.calc_shapley_values_between')
    def test_countershapley_values(self, mock_calc_shapley_values_between):
        mock_contribution_graph = {(): 0.2, (0, 1, 2): 0.8}
        mock_calc_shapley_values_between.return_value = ((0.4, 0.2, 0.1), (0, 1, 2), mock_contribution_graph)
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        values = create_plot.countershapley_values()
        expected_values = {
            'feature_names': self.feature_names,
            'feature_values': (0.4, 0.2, 0.1),
            'feature_indices': (0, 1, 2)
        }
        self.assertEqual(values, expected_values)

    def test_constellation(self):
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        fig, ax = create_plot.constellation(save_path=None)
        plt.close(fig)

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

    def test_constellation_with_highlight_changed_feature(self):
        """Test that constellation plot correctly handles highlight_changed_feature parameter.
        
        When highlight_changed_feature is passed, only the corresponding feature change's subcoalitions
        should remain emphasized (not greyed out) in the plot.
        """
        create_plot = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, self.class_names)
        
        # Test highlighting the first changed feature (index 0 in the changed features list)
        highlight_changed_feature_idx = 0
        fig, ax = create_plot.constellation(save_path=None, highlight_changed_feature=highlight_changed_feature_idx)
        
        # Verify that the plot was created successfully
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Verify that the plot contains scatter points
        collections = ax.collections
        self.assertGreater(len(collections), 0, "Plot should contain scatter plot collections")
        
        # Check that some points are colored with LIGHT_GREY (non-highlighted features)
        # and some are colored differently (highlighted feature)
        LIGHT_GREY = "#e0e0e0"
        has_light_grey = False
        has_other_color = False
        
        for collection in collections:
            colors = collection.get_facecolors()
            for color in colors:
                # Convert RGBA to hex for comparison
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0] * 255), 
                    int(color[1] * 255), 
                    int(color[2] * 255)
                )
                if hex_color.lower() == LIGHT_GREY.lower():
                    has_light_grey = True
                elif color[3] > 0:  # Not fully transparent
                    has_other_color = True
        
        # When highlighting a feature, we expect both highlighted and non-highlighted points
        self.assertTrue(has_light_grey and has_other_color, 
                       "Plot should contain both highlighted and greyed-out points")
        
        plt.close(fig)
