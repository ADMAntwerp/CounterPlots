import unittest
import numpy as np
from counterplots.utils.approx import (
    int_to_bits,
    bits_to_feature_sets,
    insert_bit,
    get_outer_edges,
    sample_random_edges,
    get_subcoalitions_sample
)
from counterplots.utils.process import calc_contribution_graph


class TestApproxHelperFunctions(unittest.TestCase):
    """Test helper functions for the sampling algorithm."""

    def test_int_to_bits_single_integer(self):
        """Test conversion of single integer to binary array."""
        result = int_to_bits([5], width=4)
        expected = np.array([[0, 1, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result[0], expected[0])

    def test_int_to_bits_multiple_integers(self):
        """Test conversion of multiple integers to binary arrays."""
        result = int_to_bits([0, 1, 2, 3], width=2)
        expected = [
            np.array([0, 0], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
            np.array([1, 0], dtype=np.uint8),
            np.array([1, 1], dtype=np.uint8),
        ]
        for i, r in enumerate(result):
            np.testing.assert_array_equal(r, expected[i])

    def test_int_to_bits_width_parameter(self):
        """Test that width parameter controls output size."""
        result = int_to_bits([7], width=3)
        self.assertEqual(len(result[0]), 3)
        np.testing.assert_array_equal(result[0], np.array([1, 1, 1], dtype=np.uint8))

    def test_bits_to_feature_sets_simple(self):
        """Test conversion of binary arrays to feature sets."""
        bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        features = ['A', 'B', 'C']
        result = bits_to_feature_sets(bits, features)
        self.assertEqual(result[0], ('A', 'C'))
        self.assertEqual(result[1], ('B', 'C'))

    def test_bits_to_feature_sets_empty_set(self):
        """Test that all zeros produces empty tuple."""
        bits = np.array([[0, 0, 0]], dtype=np.uint8)
        features = ['A', 'B', 'C']
        result = bits_to_feature_sets(bits, features)
        self.assertEqual(result[0], ())

    def test_bits_to_feature_sets_full_set(self):
        """Test that all ones produces full feature set."""
        bits = np.array([[1, 1, 1]], dtype=np.uint8)
        features = ['A', 'B', 'C']
        result = bits_to_feature_sets(bits, features)
        self.assertEqual(result[0], ('A', 'B', 'C'))

    def test_insert_bit_at_beginning(self):
        """Test inserting bit at position 0."""
        bits = np.array([[0, 1]], dtype=np.uint8)
        result = insert_bit(bits, 0, 1)
        expected = np.array([[1, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_insert_bit_at_middle(self):
        """Test inserting bit at middle position."""
        bits = np.array([[0, 1]], dtype=np.uint8)
        result = insert_bit(bits, 1, 1)
        expected = np.array([[0, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_insert_bit_at_end(self):
        """Test inserting bit at end position."""
        bits = np.array([[0, 1]], dtype=np.uint8)
        result = insert_bit(bits, 2, 1)
        expected = np.array([[0, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)


class TestOuterEdges(unittest.TestCase):
    """Test the get_outer_edges function."""

    def test_get_outer_edges_three_features(self):
        """Test outer edges for 3 features."""
        features = ['A', 'B', 'C']
        start, end = get_outer_edges(features)
        
        # Should have 2 * n_features edges (interleaved)
        self.assertEqual(len(start), 6)
        self.assertEqual(len(end), 6)
        
        # Edges are interleaved: () -> singleton, all-but-one -> full
        full_set = ('A', 'B', 'C')
        singleton_count = 0
        full_count = 0
        
        for i in range(6):
            if start[i] == ():
                # Empty → singleton edge
                singleton_count += 1
                self.assertEqual(len(end[i]), 1)
                self.assertIn(end[i][0], features)
            else:
                # All-but-one → full edge
                full_count += 1
                self.assertEqual(len(start[i]), 2)  # all-but-one has n-1 features
                self.assertEqual(end[i], full_set)
        
        # Should have n of each type
        self.assertEqual(singleton_count, 3)
        self.assertEqual(full_count, 3)

    def test_get_outer_edges_two_features(self):
        """Test outer edges for 2 features."""
        features = ['X', 'Y']
        start, end = get_outer_edges(features)
        
        self.assertEqual(len(start), 4)
        self.assertEqual(len(end), 4)
        
        # Edges are interleaved
        full_set = ('X', 'Y')
        empty_to_singleton = []
        almost_full_to_full = []
        
        for i in range(4):
            if start[i] == ():
                empty_to_singleton.append(end[i])
            else:
                almost_full_to_full.append((start[i], end[i]))
        
        # Should have 2 empty → singleton edges
        self.assertEqual(len(empty_to_singleton), 2)
        self.assertIn(('X',), empty_to_singleton)
        self.assertIn(('Y',), empty_to_singleton)
        
        # Should have 2 all-but-one → full edges
        self.assertEqual(len(almost_full_to_full), 2)
        for s, e in almost_full_to_full:
            self.assertEqual(len(s), 1)
            self.assertEqual(e, full_set)


class TestSampleRandomEdges(unittest.TestCase):
    """Test the sample_random_edges function."""

    def test_sample_random_edges_basic(self):
        """Test basic functionality of random edge sampling."""
        features = ['A', 'B', 'C']
        sample_size = 9
        rng = np.random.default_rng(42)
        
        start, end = sample_random_edges(features, sample_size, rng)
        
        # Should return approximately sample_size edges (evenly distributed)
        self.assertGreaterEqual(len(start), sample_size // len(features))
        self.assertEqual(len(start), len(end))

    def test_sample_random_edges_even_distribution(self):
        """Test that samples are evenly distributed across features."""
        features = ['A', 'B', 'C', 'D']
        sample_size = 16
        rng = np.random.default_rng(42)
        
        start, end = sample_random_edges(features, sample_size, rng)
        
        # Should have samples_per_feat * n_features edges
        expected_per_feat = sample_size // len(features)
        self.assertEqual(len(start), expected_per_feat * len(features))

    def test_sample_random_edges_minimum_sample_size(self):
        """Test that minimum sample size equals number of features."""
        features = ['A', 'B', 'C']
        sample_size = 3
        rng = np.random.default_rng(42)
        
        start, end = sample_random_edges(features, sample_size, rng)
        self.assertEqual(len(start), 3)

    def test_sample_random_edges_small_sample_size_error(self):
        """Test that sample size less than n_features raises assertion."""
        features = ['A', 'B', 'C']
        sample_size = 2
        rng = np.random.default_rng(42)
        
        with self.assertRaises(AssertionError):
            sample_random_edges(features, sample_size, rng)

    def test_sample_random_edges_edges_differ_by_one_feature(self):
        """Test that each edge connects coalitions differing by one feature."""
        features = ['A', 'B']
        sample_size = 4
        rng = np.random.default_rng(42)
        
        start, end = sample_random_edges(features, sample_size, rng)
        
        for s, e in zip(start, end):
            # Convert to sets to find difference
            s_set = set(s)
            e_set = set(e)
            # End should have exactly one more feature than start
            self.assertEqual(len(e_set - s_set), 1)
            self.assertEqual(len(s_set - e_set), 0)

    def test_sample_random_edges_reproducibility(self):
        """Test that same seed produces same results."""
        features = ['A', 'B', 'C']
        sample_size = 9
        
        rng1 = np.random.default_rng(123)
        start1, end1 = sample_random_edges(features, sample_size, rng1)
        
        rng2 = np.random.default_rng(123)
        start2, end2 = sample_random_edges(features, sample_size, rng2)
        
        self.assertEqual(start1, start2)
        self.assertEqual(end1, end2)


class TestGetSubcoalitionsSample(unittest.TestCase):
    """Test the get_subcoalitions_sample function."""

    def test_get_subcoalitions_sample_basic(self):
        """Test basic functionality of subcoalition sampling."""
        features = ['A', 'B', 'C']
        sample_size = 6
        rng = np.random.default_rng(42)
        
        result = get_subcoalitions_sample(features, sample_size, rng)
        
        # Should return a tuple of unique coalitions
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), len(set(result)))  # All unique

    def test_get_subcoalitions_sample_contains_outer_nodes(self):
        """Test that outer nodes are always included when ensure_outer_edges=True."""
        features = ['A', 'B', 'C']
        sample_size = 6
        rng = np.random.default_rng(42)
        
        result = get_subcoalitions_sample(
            features, sample_size, rng, ensure_outer_edges=True
        )
        
        # Should contain empty set and full set
        self.assertIn((), result)
        self.assertIn(tuple(features), result)

    def test_get_subcoalitions_sample_without_outer_edges(self):
        """Test sampling without ensuring all outer edges."""
        features = ['A', 'B']
        sample_size = 2
        rng = np.random.default_rng(42)
        
        result = get_subcoalitions_sample(
            features, sample_size, rng, ensure_outer_edges=False
        )
        
        # Should still contain at least empty and full set
        self.assertIn((), result)
        self.assertIn(tuple(features), result)

    def test_get_subcoalitions_sample_fractional_size(self):
        """Test that fractional sample size is converted to integer."""
        features = ['A', 'B', 'C']
        sample_size = 0.5  # 50% of max edges
        rng = np.random.default_rng(42)
        
        result = get_subcoalitions_sample(features, sample_size, rng)
        
        # Should work without error and return some coalitions
        self.assertGreater(len(result), 0)

    def test_get_subcoalitions_sample_return_as_edges(self):
        """Test returning results as edges instead of nodes."""
        features = ['A', 'B']
        sample_size = 4
        rng = np.random.default_rng(42)
        
        start, end = get_subcoalitions_sample(
            features, sample_size, rng, return_as_edges=True
        )
        
        # Should return two tuples
        self.assertIsInstance(start, tuple)
        self.assertIsInstance(end, tuple)
        self.assertEqual(len(start), len(end))

    def test_get_subcoalitions_sample_edges_are_unique(self):
        """Test that returned edges are deduplicated."""
        features = ['A', 'B']
        sample_size = 10
        rng = np.random.default_rng(42)
        
        start, end = get_subcoalitions_sample(
            features, sample_size, rng, return_as_edges=True
        )
        
        edges = list(zip(start, end))
        # All edges should be unique
        self.assertEqual(len(edges), len(set(edges)))

    def test_get_subcoalitions_sample_default_rng(self):
        """Test that function works without providing rng."""
        features = ['A', 'B', 'C']
        sample_size = 6
        
        result = get_subcoalitions_sample(features, sample_size)
        
        # Should work without error
        self.assertGreater(len(result), 0)

    def test_get_subcoalitions_sample_large_sample(self):
        """Test sampling with large sample size."""
        features = ['A', 'B', 'C', 'D']
        # Max edges is n_features * 2^(n_features - 1) = 4 * 8 = 32
        sample_size = 30
        rng = np.random.default_rng(42)
        
        result = get_subcoalitions_sample(features, sample_size, rng)
        
        # Should return valid coalitions
        self.assertGreater(len(result), 0)
        # Each coalition should only contain features from input
        for coalition in result:
            for feature in coalition:
                self.assertIn(feature, features)

    def test_get_subcoalitions_sample_integer_features(self):
        """Test that function works with integer features."""
        features = [0, 1, 2]
        sample_size = 6
        rng = np.random.default_rng(42)
        
        result = get_subcoalitions_sample(features, sample_size, rng)
        
        # Should work with integer features
        self.assertGreater(len(result), 0)
        self.assertIn((), result)
        self.assertIn((0, 1, 2), result)


class TestSampleKwargsIntegration(unittest.TestCase):
    """Test sample_kwargs parameter handling in calc_contribution_graph."""

    def setUp(self):
        """Set up a simple test model."""
        def simple_model(data):
            """Simple linear model for testing."""
            return np.array([np.sum(x) for x in data])
        
        self.model = simple_model
        self.instance1 = np.array([0.0, 0.0, 0.0])
        self.instance2 = np.array([1.0, 1.0, 1.0])

    def test_calc_contribution_graph_with_sample_kwargs(self):
        """Test that calc_contribution_graph accepts sample_kwargs."""
        indices = [0, 1, 2]
        sample_kwargs = {'n': 6}
        
        graph = calc_contribution_graph(
            self.instance1,
            self.instance2,
            self.model,
            indices,
            sample_kwargs=sample_kwargs
        )
        
        # Should return a valid contribution graph
        self.assertIsInstance(graph, dict)
        self.assertGreater(len(graph), 0)
        
        # Should contain outer nodes
        self.assertIn((), graph)
        self.assertIn((0, 1, 2), graph)

    def test_calc_contribution_graph_sample_kwargs_missing_n(self):
        """Test that missing 'n' key raises ValueError."""
        indices = [0, 1, 2]
        sample_kwargs = {'ensure_outer_edges': True}  # Missing 'n'
        
        with self.assertRaises(ValueError) as cm:
            calc_contribution_graph(
                self.instance1,
                self.instance2,
                self.model,
                indices,
                sample_kwargs=sample_kwargs
            )
        
        self.assertIn("must contain 'n' key", str(cm.exception))

    def test_calc_contribution_graph_sample_kwargs_with_ensure_outer_edges(self):
        """Test sample_kwargs with ensure_outer_edges parameter."""
        indices = [0, 1]
        sample_kwargs = {'n': 4, 'ensure_outer_edges': True}
        
        graph = calc_contribution_graph(
            self.instance1[:2],
            self.instance2[:2],
            self.model,
            indices,
            sample_kwargs=sample_kwargs
        )
        
        # Should include all outer edges
        self.assertIn((), graph)
        self.assertIn((0,), graph)
        self.assertIn((1,), graph)
        self.assertIn((0, 1), graph)

    def test_calc_contribution_graph_sample_kwargs_without_ensure_outer_edges(self):
        """Test sample_kwargs with ensure_outer_edges=False."""
        indices = [0, 1]
        sample_kwargs = {'n': 4, 'ensure_outer_edges': False}
        
        graph = calc_contribution_graph(
            self.instance1[:2],
            self.instance2[:2],
            self.model,
            indices,
            sample_kwargs=sample_kwargs
        )
        
        # Should still have at least empty and full sets
        self.assertIn((), graph)
        self.assertIn((0, 1), graph)

    def test_calc_contribution_graph_without_sample_kwargs(self):
        """Test that calc_contribution_graph works without sample_kwargs (full enumeration)."""
        indices = [0, 1]
        
        graph = calc_contribution_graph(
            self.instance1[:2],
            self.instance2[:2],
            self.model,
            indices,
            sample_kwargs=None
        )
        
        # Should enumerate all 2^2 = 4 subcoalitions
        self.assertEqual(len(graph), 4)
        self.assertIn((), graph)
        self.assertIn((0,), graph)
        self.assertIn((1,), graph)
        self.assertIn((0, 1), graph)

    def test_calc_contribution_graph_sample_reduces_coalitions(self):
        """Test that sampling reduces the number of coalitions evaluated."""
        # Use 4 features so we have more coalitions to sample from
        instance1 = np.array([0.0, 0.0, 0.0, 0.0])
        instance2 = np.array([1.0, 1.0, 1.0, 1.0])
        indices = [0, 1, 2, 3]
        
        # Full enumeration: 2^4 = 16 coalitions
        full_graph = calc_contribution_graph(
            instance1,
            instance2,
            self.model,
            indices,
            sample_kwargs=None
        )
        
        # Sampling with small sample size and no outer edges requirement
        sample_kwargs = {'n': 4, 'ensure_outer_edges': False}
        sampled_graph = calc_contribution_graph(
            instance1,
            instance2,
            self.model,
            indices,
            sample_kwargs=sample_kwargs
        )
        
        # Full graph should have all 16 coalitions
        self.assertEqual(len(full_graph), 16)
        # Sampled graph should have fewer coalitions
        self.assertLess(len(sampled_graph), len(full_graph))

    def test_calc_contribution_graph_sample_predictions_are_correct(self):
        """Test that sampled graph contains correct predictions."""
        indices = [0, 1]
        sample_kwargs = {'n': 4}
        
        graph = calc_contribution_graph(
            self.instance1[:2],
            self.instance2[:2],
            self.model,
            indices,
            sample_kwargs=sample_kwargs
        )
        
        # Verify predictions match expected values
        self.assertEqual(graph[()], 0.0)  # [0, 0] -> 0
        self.assertEqual(graph[(0, 1)], 2.0)  # [1, 1] -> 2


if __name__ == '__main__':
    unittest.main()
