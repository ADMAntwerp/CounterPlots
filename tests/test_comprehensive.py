"""
Comprehensive unit tests for the CounterPlots package.

Covers main cases and edge cases for:
- CreatePlot class (initialization, greedy, countershapley, constellation)
- Verification utilities
- Process utilities (adapted model, greedy strategy, Shapley values, contribution graph)
- Approx utilities (bit operations, edge sampling, subcoalition sampling)
- Plotting functions
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI
import matplotlib.pyplot as plt  # noqa: E402 — must be after backend set

from counterplots import CreatePlot
from counterplots.utils.verification import (
    verify_shape,
    verify_factual_counterfactual_shape,
    verify_factual_cf,
    verify_feature_names,
    verify_model_prediction,
    verify_binary_class,
    verify_class_names,
)
from counterplots.utils.process import (
    AdaptedModel,
    create_adapted_model_and_class,
    greedy_strategy,
    calc_adapted_instance,
    calc_contribution_graph,
    calc_shapley_values_between,
    get_shapley_feature_contribution,
    _convert_legendvalue,
    _format_e,
    _get_all_subcoalitions,
    _generate_subcoalitions,
)
from counterplots.utils.approx import (
    int_to_bits,
    bits_to_feature_sets,
    insert_bit,
    get_outer_edges,
    sample_random_edges,
    _get_subcoalitions_sample,
)
from counterplots.utils.plots import (
    make_greedy_plot,
    make_countershapley_plot,
    make_constellation_plot,
)


# ---------------------------------------------------------------------------
# Helper: a deterministic mock model used across multiple test classes
# ---------------------------------------------------------------------------
def _mock_model_3feat(data):
    """A deterministic model for 3 features [0.1, 0.2, 0.3] <-> [0.4, 0.5, 0.6]."""
    lookup = {
        (0.1, 0.2, 0.3): 0.0,
        (0.4, 0.5, 0.6): 1.0,
        (0.1, 0.5, 0.6): 0.4,
        (0.4, 0.2, 0.6): 0.2,
        (0.4, 0.5, 0.3): 0.3,
        (0.1, 0.2, 0.6): 0.25,
        (0.4, 0.2, 0.3): 0.4,
        (0.1, 0.5, 0.3): 0.44,
    }
    return np.array([lookup[tuple(x)] for x in data])


def _simple_sum_model(data):
    """Model that returns sum of features, clipped to [0,1]."""
    return np.array([min(max(np.sum(x), 0), 1) for x in data])


def _binary_model(data):
    """Simple binary model: returns 0 if sum < 10, else 1."""
    return np.array([
        np.array([0]) if sum(r) < 10 else np.array([1])
        for r in data
    ])


def _proba_model(data):
    """Model returning 2-column predict_proba style output."""
    out = []
    for x in data:
        s = np.sum(x) / 15.0
        s = min(max(s, 0), 1)
        out.append([1 - s, s])
    return np.array(out)


# ===========================================================================
# 1. VERIFICATION TESTS
# ===========================================================================
class TestVerifyShape(unittest.TestCase):
    """Tests for verify_shape."""

    def test_valid_1d(self):
        verify_shape(np.array([1, 2, 3]))

    def test_invalid_2d(self):
        with self.assertRaises(ValueError):
            verify_shape(np.array([[1, 2]]))

    def test_invalid_0d(self):
        # A scalar wrapped in array is 0-d
        with self.assertRaises(ValueError):
            verify_shape(np.array(5))


class TestVerifyFactualCounterfactualShape(unittest.TestCase):
    """Tests for verify_factual_counterfactual_shape."""

    def test_valid(self):
        verify_factual_counterfactual_shape(
            np.array([1, 2, 3]), np.array([4, 5, 6])
        )

    def test_factual_2d(self):
        with self.assertRaises(ValueError):
            verify_factual_counterfactual_shape(
                np.array([[1, 2]]), np.array([4, 5])
            )

    def test_cf_2d(self):
        with self.assertRaises(ValueError):
            verify_factual_counterfactual_shape(
                np.array([1, 2]), np.array([[4, 5]])
            )


class TestVerifyFactualCf(unittest.TestCase):
    """Tests for verify_factual_cf."""

    def test_valid_same_shape(self):
        verify_factual_cf(np.array([1, 2]), np.array([3, 4]))

    def test_different_lengths(self):
        with self.assertRaises(ValueError):
            verify_factual_cf(np.array([1, 2, 3]), np.array([4, 5]))


class TestVerifyFeatureNames(unittest.TestCase):
    """Tests for verify_feature_names."""

    def test_valid(self):
        verify_feature_names(np.array([1, 2, 3]), ['a', 'b', 'c'])

    def test_too_few_names(self):
        with self.assertRaises(ValueError):
            verify_feature_names(np.array([1, 2, 3]), ['a', 'b'])

    def test_too_many_names(self):
        with self.assertRaises(ValueError):
            verify_feature_names(np.array([1, 2]), ['a', 'b', 'c'])


class TestVerifyModelPrediction(unittest.TestCase):
    """Tests for verify_model_prediction."""

    def test_valid_callable(self):
        verify_model_prediction(lambda x: np.array([0.5]), np.array([1, 2]))

    def test_non_callable(self):
        with self.assertRaises(ValueError):
            verify_model_prediction("not_callable", np.array([1, 2]))


class TestVerifyBinaryClass(unittest.TestCase):
    """Tests for verify_binary_class."""

    def test_1d_output(self):
        model = lambda x: np.array([0.5])
        verify_binary_class(model, np.array([1, 2]))

    def test_2d_binary(self):
        model = lambda x: np.array([[0.3, 0.7]])
        verify_binary_class(model, np.array([1, 2]))

    def test_multiclass_raises(self):
        model = lambda x: np.array([[0.1, 0.5, 0.4]])
        with self.assertRaises(ValueError):
            verify_binary_class(model, np.array([1, 2]))

    def test_3d_raises(self):
        model = lambda x: np.array([[[1]]])
        with self.assertRaises(ValueError):
            verify_binary_class(model, np.array([1, 2]))


class TestVerifyClassNames(unittest.TestCase):
    """Tests for verify_class_names."""

    def test_valid_dict(self):
        verify_class_names({0: 'A', 1: 'B'})

    def test_string_raises(self):
        with self.assertRaises(ValueError):
            verify_class_names("bad")

    def test_list_raises(self):
        with self.assertRaises(ValueError):
            verify_class_names(['A', 'B'])

    def test_three_classes_raises(self):
        with self.assertRaises(ValueError):
            verify_class_names({0: 'A', 1: 'B', 2: 'C'})

    def test_one_class_raises(self):
        with self.assertRaises(ValueError):
            verify_class_names({0: 'A'})


# ===========================================================================
# 2. PROCESS TESTS
# ===========================================================================
class TestAdaptedModel(unittest.TestCase):
    """Tests for the AdaptedModel wrapper."""

    def test_no_col_no_invert(self):
        base = lambda x: np.array([0.3])
        model = AdaptedModel(base, col=None, invert=False)
        result = model(np.array([[1, 2]]))
        self.assertAlmostEqual(result[0], 0.3)

    def test_no_col_invert(self):
        base = lambda x: np.array([0.3])
        model = AdaptedModel(base, col=None, invert=True)
        result = model(np.array([[1, 2]]))
        self.assertAlmostEqual(result[0], 0.7)

    def test_with_col(self):
        base = lambda x: np.array([[0.3, 0.7]])
        model = AdaptedModel(base, col=1, invert=False)
        result = model(np.array([[1, 2]]))
        self.assertAlmostEqual(result[0], 0.7)

    def test_with_col_and_invert(self):
        base = lambda x: np.array([[0.3, 0.7]])
        model = AdaptedModel(base, col=1, invert=True)
        result = model(np.array([[1, 2]]))
        self.assertAlmostEqual(result[0], 0.3)


class TestCreateAdaptedModelAndClass(unittest.TestCase):
    """Tests for create_adapted_model_and_class."""

    def test_1d_low_factual(self):
        """Factual prob < 0.5 → no invert."""
        model = _mock_model_3feat
        factual = np.array([0.1, 0.2, 0.3])  # score=0.0
        adapted, classes = create_adapted_model_and_class(
            factual, model, {0: 'A', 1: 'B'}
        )
        self.assertTrue(callable(adapted))
        self.assertEqual(classes[0], 'A')
        self.assertEqual(classes[1], 'B')

    def test_1d_high_factual(self):
        """Factual prob >= 0.5 → invert."""
        model = _mock_model_3feat
        factual = np.array([0.4, 0.5, 0.6])  # score=1.0
        adapted, classes = create_adapted_model_and_class(
            factual, model, {0: 'A', 1: 'B'}
        )
        self.assertEqual(classes[0], 'B')
        self.assertEqual(classes[1], 'A')

    def test_2d_proba(self):
        """Model returns 2-column output."""
        adapted, classes = create_adapted_model_and_class(
            np.array([1, 2, 3]), _proba_model, {0: 'Neg', 1: 'Pos'}
        )
        self.assertTrue(callable(adapted))


class TestGreedyStrategy(unittest.TestCase):
    """Tests for greedy_strategy."""

    def test_basic(self):
        factual = np.array([0.1, 0.2, 0.3])
        cf = np.array([0.4, 0.5, 0.6])
        features_data = greedy_strategy(
            factual, cf, _mock_model_3feat, ['X', 'Y', 'Z']
        )
        self.assertEqual(len(features_data), 3)
        # Each entry should have required keys
        for fd in features_data:
            self.assertIn('score', fd)
            self.assertIn('name', fd)
            self.assertIn('factual', fd)
            self.assertIn('counterfactual', fd)

    def test_greedy_order(self):
        """First feature should have highest single-step score."""
        factual = np.array([0.1, 0.2, 0.3])
        cf = np.array([0.4, 0.5, 0.6])
        features_data = greedy_strategy(
            factual, cf, _mock_model_3feat, ['X', 'Y', 'Z']
        )
        # The greedy first pick should be Y (score=0.44 is highest single change)
        self.assertEqual(features_data[0]['name'], 'Y')
        self.assertAlmostEqual(features_data[0]['score'], 0.44)

    def test_all_features_appear(self):
        factual = np.array([0.1, 0.2, 0.3])
        cf = np.array([0.4, 0.5, 0.6])
        features_data = greedy_strategy(
            factual, cf, _mock_model_3feat, ['X', 'Y', 'Z']
        )
        names = {fd['name'] for fd in features_data}
        self.assertEqual(names, {'X', 'Y', 'Z'})


class TestCalcAdaptedInstance(unittest.TestCase):
    """Tests for calc_adapted_instance."""

    def test_basic(self):
        instance = np.array([0.1, 0.2, 0.3])
        instance2 = np.array([0.4, 0.5, 0.6])
        result = calc_adapted_instance(instance, instance2, [0, 2])
        np.testing.assert_array_almost_equal(result, [0.4, 0.2, 0.6])

    def test_no_indices(self):
        instance = np.array([1, 2, 3])
        instance2 = np.array([4, 5, 6])
        result = calc_adapted_instance(instance, instance2, [])
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_all_indices(self):
        instance = np.array([1, 2, 3])
        instance2 = np.array([4, 5, 6])
        result = calc_adapted_instance(instance, instance2, [0, 1, 2])
        np.testing.assert_array_equal(result, [4, 5, 6])


class TestConvertLegendvalue(unittest.TestCase):
    """Tests for _convert_legendvalue."""

    def test_large_integer(self):
        self.assertEqual(_convert_legendvalue(10000), '1.00E4')

    def test_small_decimal(self):
        self.assertEqual(_convert_legendvalue(0.00001), '1.00E-5')

    def test_normal_decimal(self):
        self.assertEqual(_convert_legendvalue(0.123456), 0.12346)

    def test_string(self):
        self.assertEqual(_convert_legendvalue('ABC'), 'ABC')

    def test_integer_below_threshold(self):
        # 9999 is not > 9999, so it should return as-is
        self.assertEqual(_convert_legendvalue(9999), 9999)

    def test_zero(self):
        self.assertEqual(_convert_legendvalue(0), 0)

    def test_negative_small(self):
        result = _convert_legendvalue(-0.000001)
        self.assertEqual(result, '-1.00E-6')

    def test_normal_negative(self):
        result = _convert_legendvalue(-3.14)
        self.assertEqual(result, -3.14)


class TestFormatE(unittest.TestCase):
    """Tests for _format_e."""

    def test_positive_exponent(self):
        self.assertEqual(_format_e(10000), '1.00E4')

    def test_negative_exponent(self):
        self.assertEqual(_format_e(0.001), '1.00E-3')


class TestGetAllSubcoalitions(unittest.TestCase):
    """Tests for _get_all_subcoalitions."""

    def test_two_features(self):
        result = _get_all_subcoalitions([0, 1])
        self.assertEqual(len(result), 4)  # 2^2
        self.assertIn((), result)
        self.assertIn((0,), result)
        self.assertIn((1,), result)
        self.assertIn((0, 1), result)

    def test_three_features(self):
        result = _get_all_subcoalitions([0, 1, 2])
        self.assertEqual(len(result), 8)  # 2^3

    def test_empty(self):
        result = _get_all_subcoalitions([])
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [()])


class TestCalcContributionGraph(unittest.TestCase):
    """Tests for calc_contribution_graph."""

    def test_basic(self):
        instance1 = np.array([0.0, 0.0])
        instance2 = np.array([1.0, 1.0])
        model = lambda x: np.array([np.sum(row) for row in x])
        graph = calc_contribution_graph(
            instance1, instance2, model,
            class_of_interest_index=0,
            multiproc_kwargs={"min_size": 100, "cpu_count": 1}
        )
        self.assertIn((), graph)
        self.assertIn((0,), graph)
        self.assertIn((1,), graph)
        self.assertIn((0, 1), graph)
        self.assertEqual(len(graph), 4)

    def test_predictions_correct(self):
        instance1 = np.array([0.0, 0.0])
        instance2 = np.array([1.0, 1.0])
        model = lambda x: np.array([np.sum(row) for row in x])
        graph = calc_contribution_graph(
            instance1, instance2, model,
            class_of_interest_index=0,
            multiproc_kwargs={"min_size": 100, "cpu_count": 1}
        )
        self.assertAlmostEqual(graph[()], 0.0)
        self.assertAlmostEqual(graph[(0,)], 1.0)
        self.assertAlmostEqual(graph[(1,)], 1.0)
        self.assertAlmostEqual(graph[(0, 1)], 2.0)

    def test_with_sample_kwargs(self):
        instance1 = np.array([0.0, 0.0, 0.0])
        instance2 = np.array([1.0, 1.0, 1.0])
        model = lambda x: np.array([np.sum(row) for row in x])
        graph = calc_contribution_graph(
            instance1, instance2, model,
            sample_kwargs={'n': 6},
            multiproc_kwargs={"min_size": 100, "cpu_count": 1}
        )
        self.assertIsInstance(graph, dict)
        self.assertIn((), graph)
        self.assertIn((0, 1, 2), graph)


class TestGetShapleyFeatureContribution(unittest.TestCase):
    """Tests for get_shapley_feature_contribution."""

    def test_symmetric_features(self):
        """Two identical features should have equal Shapley values."""
        graph = {
            (): 0.0,
            (0,): 1.0,
            (1,): 1.0,
            (0, 1): 2.0,
        }
        c0 = get_shapley_feature_contribution(graph, 0)
        c1 = get_shapley_feature_contribution(graph, 1)
        self.assertAlmostEqual(c0, c1)
        self.assertAlmostEqual(c0, 1.0)

    def test_sum_equals_grand_coalition(self):
        """Sum of Shapley values should equal v(N) - v(empty)."""
        graph = {
            (): 0.0,
            (0,): 0.4,
            (1,): 0.3,
            (0, 1): 1.0,
        }
        c0 = get_shapley_feature_contribution(graph, 0)
        c1 = get_shapley_feature_contribution(graph, 1)
        self.assertAlmostEqual(c0 + c1, 1.0)


class TestCalcShapleyValuesBetween(unittest.TestCase):
    """Tests for calc_shapley_values_between."""

    def test_returns_three_elements(self):
        instance1 = np.array([0.1, 0.2, 0.3])
        instance2 = np.array([0.4, 0.5, 0.6])
        contributions, indices, graph = calc_shapley_values_between(
            instance1, instance2, _mock_model_3feat,
            multiproc_kwargs={"min_size": 100, "cpu_count": 1}
        )
        self.assertEqual(len(contributions), 3)
        self.assertEqual(len(indices), 3)
        self.assertIsInstance(graph, dict)

    def test_sorted_descending(self):
        """Contributions should be sorted from largest to smallest."""
        instance1 = np.array([0.1, 0.2, 0.3])
        instance2 = np.array([0.4, 0.5, 0.6])
        contributions, indices, graph = calc_shapley_values_between(
            instance1, instance2, _mock_model_3feat,
            multiproc_kwargs={"min_size": 100, "cpu_count": 1}
        )
        for i in range(len(contributions) - 1):
            self.assertGreaterEqual(contributions[i], contributions[i + 1])

    def test_sum_equals_difference(self):
        """Sum of Shapley values should equal v(cf) - v(factual)."""
        instance1 = np.array([0.1, 0.2, 0.3])
        instance2 = np.array([0.4, 0.5, 0.6])
        contributions, indices, graph = calc_shapley_values_between(
            instance1, instance2, _mock_model_3feat,
            multiproc_kwargs={"min_size": 100, "cpu_count": 1}
        )
        total = sum(contributions)
        expected = graph[tuple(np.where(instance1 != instance2)[0])] - graph[()]
        self.assertAlmostEqual(total, expected, places=5)


# ===========================================================================
# 3. APPROX TESTS
# ===========================================================================
class TestIntToBits(unittest.TestCase):
    """Tests for int_to_bits."""

    def test_single(self):
        result = int_to_bits([5], width=4)
        np.testing.assert_array_equal(result[0], [0, 1, 0, 1])

    def test_zero(self):
        result = int_to_bits([0], width=3)
        np.testing.assert_array_equal(result[0], [0, 0, 0])

    def test_all_ones(self):
        result = int_to_bits([7], width=3)
        np.testing.assert_array_equal(result[0], [1, 1, 1])

    def test_multiple(self):
        result = int_to_bits([0, 1, 2, 3], width=2)
        np.testing.assert_array_equal(result[0], [0, 0])
        np.testing.assert_array_equal(result[1], [0, 1])
        np.testing.assert_array_equal(result[2], [1, 0])
        np.testing.assert_array_equal(result[3], [1, 1])


class TestBitsToFeatureSets(unittest.TestCase):
    """Tests for bits_to_feature_sets."""

    def test_basic(self):
        bits = np.array([[1, 0, 1]], dtype=np.uint8)
        features = ['A', 'B', 'C']
        result = bits_to_feature_sets(bits, features)
        self.assertEqual(result[0], ('A', 'C'))

    def test_empty_set(self):
        bits = np.array([[0, 0]], dtype=np.uint8)
        result = bits_to_feature_sets(bits, ['X', 'Y'])
        self.assertEqual(result[0], ())

    def test_full_set(self):
        bits = np.array([[1, 1, 1]], dtype=np.uint8)
        result = bits_to_feature_sets(bits, ['A', 'B', 'C'])
        self.assertEqual(result[0], ('A', 'B', 'C'))

    def test_integer_features(self):
        bits = np.array([[1, 0, 1]], dtype=np.uint8)
        result = bits_to_feature_sets(bits, [0, 1, 2])
        self.assertEqual(result[0], (0, 2))


class TestInsertBit(unittest.TestCase):
    """Tests for insert_bit."""

    def test_beginning(self):
        bits = np.array([[0, 1]], dtype=np.uint8)
        result = insert_bit(bits, 0, 1)
        np.testing.assert_array_equal(result, [[1, 0, 1]])

    def test_middle(self):
        bits = np.array([[0, 1]], dtype=np.uint8)
        result = insert_bit(bits, 1, 1)
        np.testing.assert_array_equal(result, [[0, 1, 1]])

    def test_end(self):
        bits = np.array([[0, 1]], dtype=np.uint8)
        result = insert_bit(bits, 2, 1)
        np.testing.assert_array_equal(result, [[0, 1, 1]])


class TestGetOuterEdges(unittest.TestCase):
    """Tests for get_outer_edges."""

    def test_two_features(self):
        start, end = get_outer_edges([0, 1])
        self.assertEqual(len(start), 4)
        self.assertEqual(len(end), 4)
        # Must contain empty→singleton edges
        for i in range(len(start)):
            if start[i] == ():
                self.assertEqual(len(end[i]), 1)

    def test_three_features(self):
        start, end = get_outer_edges(['A', 'B', 'C'])
        self.assertEqual(len(start), 6)
        self.assertEqual(len(end), 6)
        full = ('A', 'B', 'C')
        full_count = sum(1 for e in end if e == full)
        self.assertEqual(full_count, 3)


class TestSampleRandomEdges(unittest.TestCase):
    """Tests for sample_random_edges."""

    def test_basic(self):
        rng = np.random.default_rng(42)
        start, end = sample_random_edges(['A', 'B', 'C'], 9, rng)
        self.assertEqual(len(start), len(end))
        self.assertGreater(len(start), 0)

    def test_edges_differ_by_one_feature(self):
        rng = np.random.default_rng(42)
        start, end = sample_random_edges(['A', 'B'], 4, rng)
        for s, e in zip(start, end):
            diff = set(e) - set(s)
            self.assertEqual(len(diff), 1)
            self.assertEqual(len(set(s) - set(e)), 0)

    def test_too_small_sample_raises(self):
        with self.assertRaises(AssertionError):
            sample_random_edges(['A', 'B', 'C'], 2, np.random.default_rng(42))

    def test_reproducibility(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        s1, e1 = sample_random_edges(['A', 'B'], 4, rng1)
        s2, e2 = sample_random_edges(['A', 'B'], 4, rng2)
        self.assertEqual(s1, s2)
        self.assertEqual(e1, e2)


class TestGetSubcoalitionsSample(unittest.TestCase):
    """Tests for _get_subcoalitions_sample."""

    def test_basic(self):
        rng = np.random.default_rng(42)
        result = _get_subcoalitions_sample([0, 1, 2], sample_size=6, rng=rng)
        self.assertIsInstance(result, tuple)
        # All unique
        self.assertEqual(len(result), len(set(result)))

    def test_contains_outer_nodes(self):
        rng = np.random.default_rng(42)
        result = _get_subcoalitions_sample(
            [0, 1, 2], sample_size=6, rng=rng, ensure_outer_edges=True
        )
        self.assertIn((), result)
        self.assertIn((0, 1, 2), result)

    def test_without_outer_edges(self):
        rng = np.random.default_rng(42)
        result = _get_subcoalitions_sample(
            [0, 1], sample_size=2, rng=rng, ensure_outer_edges=False
        )
        # Should still have empty and full
        self.assertIn((), result)
        self.assertIn((0, 1), result)

    def test_fractional_sample_size(self):
        rng = np.random.default_rng(42)
        result = _get_subcoalitions_sample([0, 1, 2], sample_size=0.5, rng=rng)
        self.assertGreater(len(result), 0)

    def test_integer_features(self):
        rng = np.random.default_rng(42)
        result = _get_subcoalitions_sample([0, 1, 2], sample_size=6, rng=rng)
        self.assertIn((), result)
        self.assertIn((0, 1, 2), result)


# ===========================================================================
# 4. PLOTTING TESTS
# ===========================================================================
class TestMakeGreedyPlot(unittest.TestCase):
    """Tests for make_greedy_plot."""

    def test_basic(self):
        features_data = [
            {'score': 0.4, 'name': 'F1', 'factual': 1, 'counterfactual': 2},
            {'score': 0.8, 'name': 'F2', 'factual': 3, 'counterfactual': 4},
        ]
        fig, ax = make_greedy_plot(0.2, features_data, {0: 'A', 1: 'B'}, None)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)

    def test_single_feature(self):
        features_data = [
            {'score': 0.9, 'name': 'Only', 'factual': 0, 'counterfactual': 1},
        ]
        fig, ax = make_greedy_plot(0.1, features_data, {0: 'No', 1: 'Yes'}, None)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_custom_threshold(self):
        features_data = [
            {'score': 0.3, 'name': 'F1', 'factual': 0, 'counterfactual': 1},
        ]
        fig, ax = make_greedy_plot(
            0.1, features_data, {0: 'A', 1: 'B'}, None,
            decision_threshold=0.25
        )
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestMakeCountershapleyPlot(unittest.TestCase):
    """Tests for make_countershapley_plot."""

    def test_basic(self):
        features_data = [
            {'x': 60.0, 'score': 0.4, 'name': 'F1',
             'factual': 1, 'counterfactual': 2},
            {'x': 40.0, 'score': 0.7, 'name': 'F2',
             'factual': 3, 'counterfactual': 4},
        ]
        fig, ax = make_countershapley_plot(
            0.1, features_data, {0: 'A', 1: 'B'}, None
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)

    def test_negative_contribution(self):
        """Feature with negative CounterShapley value."""
        features_data = [
            {'x': 80.0, 'score': 0.6, 'name': 'F1',
             'factual': 1, 'counterfactual': 2},
            {'x': -20.0, 'score': 0.5, 'name': 'F2',
             'factual': 3, 'counterfactual': 4},
        ]
        fig, ax = make_countershapley_plot(
            0.2, features_data, {0: 'A', 1: 'B'}, None
        )
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestMakeConstellationPlot(unittest.TestCase):
    """Tests for make_constellation_plot."""

    def test_basic(self):
        single_points_chart = np.array([[0., 0.3], [1., 0.6]])
        text_features = ['F1 (0➜1)', 'F2 (2➜3)']
        multiple_points_chart = np.array(
            [[np.array([0, 1]), 0.8]], dtype=object
        )
        multiple_points_chart_y = [0.5]
        single_points = np.array([[0., 0.3], [1., 0.6]])
        point_to_pred = {0: 0.3, 1: 0.6}

        fig, ax = make_constellation_plot(
            factual_score=0.1,
            single_points_chart=single_points_chart,
            text_features=text_features,
            multiple_points_chart=multiple_points_chart,
            multiple_points_chart_y=multiple_points_chart_y,
            single_points=single_points,
            class_names={0: 'A', 1: 'B'},
            cf_score=0.9,
            point_to_pred=point_to_pred,
            save_path=None,
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)

    def test_with_highlight(self):
        single_points_chart = np.array([[0., 0.3], [1., 0.6]])
        text_features = ['F1 (0➜1)', 'F2 (2➜3)']
        multiple_points_chart = np.array(
            [[np.array([0, 1]), 0.8]], dtype=object
        )
        multiple_points_chart_y = [0.5]
        single_points = np.array([[0., 0.3], [1., 0.6]])
        point_to_pred = {0: 0.3, 1: 0.6}

        fig, ax = make_constellation_plot(
            factual_score=0.1,
            single_points_chart=single_points_chart,
            text_features=text_features,
            multiple_points_chart=multiple_points_chart,
            multiple_points_chart_y=multiple_points_chart_y,
            single_points=single_points,
            class_names={0: 'A', 1: 'B'},
            cf_score=0.9,
            point_to_pred=point_to_pred,
            save_path=None,
            highlight_changed_feature=0,
        )
        self.assertIsNotNone(fig)
        plt.close(fig)


# ===========================================================================
# 5. CREATEPLOT INTEGRATION TESTS
# ===========================================================================
class TestCreatePlotInit(unittest.TestCase):
    """Tests for CreatePlot initialization."""

    def setUp(self):
        self.factual = np.array([1, 2, 3])
        self.cf = np.array([4, 5, 6])
        self.model_pred = _binary_model
        self.feature_names = ['feat1', 'feat2', 'feat3']
        self.class_names = {0: '0', 1: '1'}

    def test_basic_init(self):
        cp = CreatePlot(
            self.factual, self.cf, self.model_pred,
            self.feature_names, self.class_names
        )
        np.testing.assert_array_equal(cp.factual, self.factual)
        np.testing.assert_array_equal(cp.cf, self.cf)
        self.assertEqual(cp.feature_names, self.feature_names)

    def test_default_feature_names(self):
        cp = CreatePlot(self.factual, self.cf, self.model_pred, None, self.class_names)
        self.assertEqual(cp.feature_names, ['f1', 'f2', 'f3'])

    def test_default_class_names(self):
        cp = CreatePlot(self.factual, self.cf, self.model_pred, self.feature_names, None)
        # class_names may be swapped by adapted model, but should be a 2-element dict
        self.assertEqual(len(cp.class_names), 2)

    def test_custom_threshold(self):
        cp = CreatePlot(
            self.factual, self.cf, self.model_pred,
            self.feature_names, self.class_names,
            decision_threshold=0.3
        )
        self.assertEqual(cp.decision_threshold, 0.3)

    def test_factual_and_cf_scores_set(self):
        cp = CreatePlot(
            self.factual, self.cf, self.model_pred,
            self.feature_names, self.class_names
        )
        self.assertIsNotNone(cp.factual_score)
        self.assertIsNotNone(cp.cf_score)

    def test_contribution_graph_set(self):
        cp = CreatePlot(
            self.factual, self.cf, self.model_pred,
            self.feature_names, self.class_names
        )
        self.assertIsInstance(cp.contribution_graph, dict)

    def test_with_proba_model(self):
        """Model returning 2-column output should work."""
        cp = CreatePlot(
            self.factual, self.cf, _proba_model,
            self.feature_names, self.class_names
        )
        self.assertIsNotNone(cp.factual_score)


class TestCreatePlotGreedy(unittest.TestCase):
    """Tests for CreatePlot.greedy()."""

    def setUp(self):
        self.cp = CreatePlot(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            _binary_model,
            ['feat1', 'feat2', 'feat3'],
            {0: '0', 1: '1'},
        )

    def test_returns_fig_and_ax(self):
        fig, ax = self.cp.greedy(save_path=None)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)


class TestCreatePlotCountershapley(unittest.TestCase):
    """Tests for CreatePlot.countershapley()."""

    def setUp(self):
        self.cp = CreatePlot(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            _binary_model,
            ['feat1', 'feat2', 'feat3'],
            {0: '0', 1: '1'},
        )

    def test_returns_fig_and_ax(self):
        fig, ax = self.cp.countershapley(save_path=None)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)

    def test_countershapley_values(self):
        values = self.cp.countershapley_values()
        self.assertIn('feature_names', values)
        self.assertIn('feature_values', values)
        self.assertIn('feature_indices', values)
        self.assertEqual(len(values['feature_names']), 3)
        self.assertEqual(len(values['feature_values']), 3)
        self.assertEqual(len(values['feature_indices']), 3)


class TestCreatePlotConstellation(unittest.TestCase):
    """Tests for CreatePlot.constellation()."""

    def test_returns_fig_and_ax(self):
        cp = CreatePlot(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            _binary_model,
            ['feat1', 'feat2', 'feat3'],
            {0: '0', 1: '1'},
        )
        fig, ax = cp.constellation(save_path=None)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)

    def test_with_highlight(self):
        cp = CreatePlot(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            _binary_model,
            ['feat1', 'feat2', 'feat3'],
            {0: '0', 1: '1'},
        )
        fig, ax = cp.constellation(save_path=None, highlight_changed_feature=0)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_single_feature_diff_falls_back_to_greedy(self):
        """When only 1 feature differs, constellation falls back to greedy."""
        cp = CreatePlot(
            np.array([1, 2, 3]),
            np.array([1, 2, 6]),
            _binary_model,
            ['feat1', 'feat2', 'feat3'],
            {0: '0', 1: '1'},
        )
        # constellation with <2 differences prints message and calls greedy
        result = cp.constellation(save_path=None)
        # Returns None when falling back to greedy
        self.assertIsNone(result)


# ===========================================================================
# 6. EDGE CASE TESTS
# ===========================================================================
class TestEdgeCaseIdenticalInstances(unittest.TestCase):
    """Edge case: factual and counterfactual are identical."""

    def test_no_differences(self):
        # When factual == cf, there are no feature differences.
        # calc_shapley_values_between will have 0 features to process
        # and zip(*sorted(zip([], []))) raises ValueError on empty sequences.
        # This is a known edge case that is not handled gracefully.
        factual = np.array([1, 2, 3])
        cf = np.array([1, 2, 3])
        model = _binary_model

        # Creating a plot with identical instances should fail because
        # zip(*sorted(zip([], []))) raises ValueError: not enough values to unpack.
        with self.assertRaisesRegex(ValueError, "not enough values to unpack"):
            CreatePlot(factual, cf, model, ['f1', 'f2', 'f3'], {0: '0', 1: '1'})


class TestEdgeCaseSingleFeature(unittest.TestCase):
    """Edge case: only one feature in the instances."""

    def test_single_feature_instances(self):
        factual = np.array([0])
        cf = np.array([1])
        model = lambda x: np.array([float(row[0]) for row in x])
        cp = CreatePlot(factual, cf, model, ['only_feat'], {0: 'A', 1: 'B'})
        self.assertEqual(len(cp.idx_feat_diff), 1)

    def test_single_feature_greedy(self):
        factual = np.array([0])
        cf = np.array([1])
        model = lambda x: np.array([float(row[0]) for row in x])
        cp = CreatePlot(factual, cf, model, ['only_feat'], {0: 'A', 1: 'B'})
        fig, ax = cp.greedy(save_path=None)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestEdgeCaseLargeValues(unittest.TestCase):
    """Edge case: large numeric values that trigger scientific notation."""

    def test_large_feature_values(self):
        factual = np.array([100000.0, 200000.0])
        cf = np.array([300000.0, 400000.0])
        model = lambda x: np.array([
            min(max(np.sum(row) / 1000000, 0), 1) for row in x
        ])
        cp = CreatePlot(factual, cf, model, ['big1', 'big2'], {0: 'Lo', 1: 'Hi'})
        fig, ax = cp.greedy(save_path=None)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestEdgeCaseVerySmallValues(unittest.TestCase):
    """Edge case: very small feature values triggering scientific notation."""

    def test_tiny_feature_values(self):
        factual = np.array([0.000001, 0.000002])
        cf = np.array([0.000003, 0.000004])
        model = lambda x: np.array([
            min(max(np.sum(row) * 100000, 0), 1) for row in x
        ])
        cp = CreatePlot(factual, cf, model, ['tiny1', 'tiny2'], {0: 'Lo', 1: 'Hi'})
        fig, ax = cp.greedy(save_path=None)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestEdgeCaseDecisionThreshold(unittest.TestCase):
    """Edge case: decision threshold at boundary values."""

    def test_threshold_zero(self):
        factual = np.array([0, 0])
        cf = np.array([1, 1])
        model = lambda x: np.array([min(max(np.sum(row) / 2, 0), 1) for row in x])
        cp = CreatePlot(factual, cf, model, ['f1', 'f2'], {0: 'A', 1: 'B'},
                        decision_threshold=0.0)
        fig, ax = cp.greedy(save_path=None)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_threshold_one(self):
        factual = np.array([0, 0])
        cf = np.array([1, 1])
        model = lambda x: np.array([min(max(np.sum(row) / 2, 0), 1) for row in x])
        cp = CreatePlot(factual, cf, model, ['f1', 'f2'], {0: 'A', 1: 'B'},
                        decision_threshold=1.0)
        fig, ax = cp.greedy(save_path=None)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestEdgeCaseHighDimensional(unittest.TestCase):
    """Edge case: more features that differ."""

    def test_five_features(self):
        factual = np.array([0, 0, 0, 0, 0])
        cf = np.array([1, 1, 1, 1, 1])
        model = lambda x: np.array([min(max(np.sum(row) / 5, 0), 1) for row in x])
        cp = CreatePlot(factual, cf, model,
                        ['f1', 'f2', 'f3', 'f4', 'f5'],
                        {0: 'Lo', 1: 'Hi'})
        fig, ax = cp.greedy(save_path=None)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_five_features_constellation(self):
        factual = np.array([0, 0, 0, 0, 0])
        cf = np.array([1, 1, 1, 1, 1])
        model = lambda x: np.array([min(max(np.sum(row) / 5, 0), 1) for row in x])
        cp = CreatePlot(factual, cf, model,
                        ['f1', 'f2', 'f3', 'f4', 'f5'],
                        {0: 'Lo', 1: 'Hi'})
        fig, ax = cp.constellation(save_path=None)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestEdgeCasePartialFeatureDiff(unittest.TestCase):
    """Edge case: some features are the same, some differ."""

    def test_partial_diff(self):
        factual = np.array([1, 2, 3, 4])
        cf = np.array([1, 5, 3, 7])
        model = lambda x: np.array([
            min(max((np.sum(row) - 10) / 10, 0), 1) for row in x
        ])
        cp = CreatePlot(
            factual, cf, model,
            ['same1', 'diff1', 'same2', 'diff2'],
            {0: 'A', 1: 'B'}
        )
        # Only 2 features differ
        self.assertEqual(len(cp.idx_feat_diff), 2)
        fig, ax = cp.constellation(save_path=None)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestEdgeCaseConvertLegendvalueEdgeCases(unittest.TestCase):
    """Edge cases for _convert_legendvalue."""

    def test_exactly_10000(self):
        # 10000 > 9999, so scientific notation
        self.assertEqual(_convert_legendvalue(10000), '1.00E4')

    def test_exactly_9999(self):
        # Not > 9999, return as-is
        self.assertEqual(_convert_legendvalue(9999), 9999)

    def test_float_with_no_decimals(self):
        # 5.0 % 1 == 0
        self.assertEqual(_convert_legendvalue(5.0), 5.0)

    def test_negative_large(self):
        result = _convert_legendvalue(-10000)
        # -10000 is an integer but xn > 9999 check is False for negatives
        # since -10000 is not > 9999
        self.assertEqual(result, -10000)

    def test_boolean_input(self):
        # bool is a subclass of int; True = 1, False = 0
        self.assertEqual(_convert_legendvalue(True), True)
        self.assertEqual(_convert_legendvalue(False), False)


class TestEdgeCaseSampleKwargs(unittest.TestCase):
    """Edge case: using sample_kwargs with CreatePlot."""

    def test_with_sampling(self):
        factual = np.array([0, 0, 0])
        cf = np.array([1, 1, 1])
        model = lambda x: np.array([min(max(np.sum(row) / 3, 0), 1) for row in x])
        cp = CreatePlot(
            factual, cf, model,
            ['f1', 'f2', 'f3'],
            {0: 'A', 1: 'B'},
            sample_kwargs={'n': 6}
        )
        self.assertIsNotNone(cp.contribution_graph)
        values = cp.countershapley_values()
        self.assertEqual(len(values['feature_names']), 3)


class TestEdgeCaseHighlightInvalidFeature(unittest.TestCase):
    """Edge case: highlight_changed_feature with invalid type."""

    def test_string_highlight_raises(self):
        """Non-integer highlight should raise AssertionError."""
        cp = CreatePlot(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            _binary_model,
            ['f1', 'f2', 'f3'],
            {0: '0', 1: '1'},
        )
        with self.assertRaises(AssertionError):
            cp.constellation(save_path=None, highlight_changed_feature="bad")

    def test_float_highlight_raises(self):
        """Float highlight should raise AssertionError."""
        cp = CreatePlot(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            _binary_model,
            ['f1', 'f2', 'f3'],
            {0: '0', 1: '1'},
        )
        with self.assertRaises(AssertionError):
            cp.constellation(save_path=None, highlight_changed_feature=1.5)

    def test_none_highlight_works(self):
        """None highlight (default) should not raise."""
        cp = CreatePlot(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            _binary_model,
            ['f1', 'f2', 'f3'],
            {0: '0', 1: '1'},
        )
        fig, ax = cp.constellation(save_path=None, highlight_changed_feature=None)
        self.assertIsNotNone(fig)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
