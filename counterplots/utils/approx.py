"""
Methods for approximating Shapley values by subsampling coalition space.

Sampling Shapley values has an interesting problem to solve. How do you subsample the possible coalitions of features, so that:
- You don't accidentally sample a coalition that has no neighbors i.e. that cannot be compared against any other coalition by adding/removing a feature
- You don't sample the same coalition twice
- Your sample is approximately evenly distributed across features, so that the accuracy per feature has an ensured lower limit.

To solve this, consider the graph representation of coalition space (i.e. ``contribution_graph``), e.g. for 4 features A, B, C, D

( )

(A)     (B)     (C)     (D)

(AB)    (AC)    (AD)    (BC)    (BD)    (CD)

(ABC)   (ABD)   (ACD)   (BCD)

(ABCD)


# How to sample coalitions that always have neighbors

The initial intuition would be to randomly sample subcoalitions. However, this gives no guarantee that each will be useful for a Shapley comparison,
or that it is unique or evenly sampled.
The second intuition is to sample edges instead e.g. (AB)-(ABC) is the edge that adds C to the coalition AB.
Each edge by definition connects two subcoalitions that differ by a single feature, and can therefore always be used for exactly one Shapley comparison.

# How to ensure uniqueness.

To sample a unique set of edges, it suffices to sample a unique set of edge begin nodes.
Each subcoalition can be represented in binary, reflecting if they do or don't contain a feature.
For features ABCD, this will be 4 bits. Subcoalition (B,D) for example does not have A, has B, not C, has D, so the binary would be 0101.
Sampling N unique begin nodes is thus the same as sampling N unique 4-bit integers.
This is at wors O(2^K), depending on how many samples you want.

# How to ensure even distribution of samples across features

This can be done by sampling N/K edges per feature, where K is the amount of features.

In bit representation, that means the Kth bit will always be 1, and the remaining bits need to be sampled at random.
So for feature C, the coalitions to iterate will have bits --1-. Simply randomly fill the -.
This is equivalent to sampling random (N-1) bit integers and inserting a 1 at the feature-corresponding location.
One caveat: 1111 is not allowed, as this is not a starting node.

Generating edges from these starting nodes then consists of taking these nodes, considering them to be end nodes, and generating a random start node.
E.g. for feature C, the end node is -1--. The start node will be -0--.
"""

import numpy as np
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


def int_to_bits(x, width):
    """Convert integer(s) to binary array of shape (..., width)."""
    return [(e >> np.arange(width - 1, -1, -1) & 1).astype(np.uint8) for e in x]


def bits_to_feature_sets(bits, features):
    """
    Convert binary array of shape (n, K) to tuples of features.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    features = np.asarray(features)

    return [
        tuple(features[row.astype(bool)])
        for row in bits
    ]


def insert_bit(bits, idx, value):
    """Insert a bit at position idx."""
    return np.insert(bits, idx, value, axis=1)


def get_outer_edges(features):
    """Get all edges of the contribution graph that connect the empty set to singletons, and the full set to all (N-1) sets
    """
    features = tuple(features)
    n = len(features)
    start = []
    end = []
    for i in range(n):
        # empty → singleton
        start.append(())
        end.append((features[i],))
        # all-but-one → full
        start.append(tuple(f for j, f in enumerate(features) if j != i))
        end.append(features)
    return start, end


def sample_random_edges(features, sample_size, rng=None):
    """
    Sample Shapley edges by evenly distributing samples across features.

    Args:
        features (np.ndarray or list): Feature identifiers.
        sample_size (int): Total number of edges to sample.
        rng (np.random.Generator, optional)

    Returns:
        start_nodes (np.ndarray): Integer-encoded start coalitions
        end_nodes (np.ndarray): Integer-encoded end coalitions
        feature_indices (np.ndarray): Feature index for each edge
    """
    if rng is None: rng = np.random.default_rng()
    n_features = len(features)
    assert sample_size >= len(features), "I need at least {} samples to evenly sample {} features. You passed {}".format(n_features, n_features, sample_size)

    samples_per_feat = sample_size // n_features
    start_nodes = []
    end_nodes = []
    feature_indices = []

    for feat_idx in range(n_features):
        # sample unique (K-1)-bit integers
        max_states = 2 ** (n_features - 1)
        n_draw = min(samples_per_feat, max_states)
        base_ints = rng.choice(
            max_states,
            size=n_draw,
            replace=False
        )
        # Convert to bit to constuct edges
        base_bits = int_to_bits(base_ints, n_features - 1)
        end_bits = insert_bit(base_bits, feat_idx, 1)
        start_bits = end_bits.copy()
        start_bits[:, feat_idx] = 0
        start_nodes.extend(bits_to_feature_sets(start_bits, features))
        end_nodes.extend(bits_to_feature_sets(end_bits, features))

    return start_nodes, end_nodes


def get_subcoalitions_sample(
    features, 
    sample_size, 
    rng=None,
    ensure_outer_edges=True,
    ):
    n_features = len(features)
    max_n_edges = n_features * 2**(n_features-1)
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(sample_size, float):
        
        sample_size  = int(sample_size * max_n_edges)
        if sample_size < len(features):
            logger.warning(f"The provided sample size {sample_size} is too small to evenly sample {n_features} features. Setting it to the minimum of {n_features} instead.")
            sample_size = n_features
    logger.debug("Making {} Shapley comparisons (max={})".format(sample_size, max_n_edges))

    nodes = []

    # random edges
    start, end = sample_random_edges(features, sample_size, rng)
    nodes.extend(start)
    nodes.extend(end)

    if ensure_outer_edges == True:
        start, end = get_outer_edges(features)
        nodes.extend(start)
        nodes.extend(end)
    else:
        # Ensure at least outer nodes are present
        nodes.extend(((), tuple(features)))

    # deduplicate nodes
    nodes = tuple(dict.fromkeys(nodes))
    return nodes
