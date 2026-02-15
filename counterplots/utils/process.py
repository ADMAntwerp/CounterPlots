from itertools import combinations
from functools import lru_cache
import numpy as np
from math import factorial
from .approx import get_subcoalitions_sample
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


def create_adapted_model_and_class(factual, model_pred, class_names):
    """ Adapts the model to return a single class, factual class is also always 0

    Args:
        factual (np.ndarray): Factual instance
        model_pred (object): Model prediction function

    Returns:
        obejct: Adapted model prediction function
    """
    pred_test = np.array(model_pred(np.array([factual])))
    if len(pred_test.shape) == 2:
        # If model has two classes, return the probability of the second class (1)
        if pred_test.shape[1] == 2:
            def adapted_model(x): return np.array(model_pred(x))[:, 1] if pred_test[0][1] < 0.5 else np.array(
                model_pred(x))[:, 0]
            adapted_class = {0: class_names[0], 1: class_names[1]} if pred_test[0][1] < 0.5 else {0: class_names[1],
                                                                                                  1: class_names[0]}
        else:
            def adapted_model(x): return np.array(model_pred(x))[:, 0] if pred_test[0][0] < 0.5 else 1 - np.array(
                model_pred(x))[:, 0]
            adapted_class = {0: class_names[0], 1: class_names[1]} if pred_test[0][0] < 0.5 else {0: class_names[1],
                                                                                                  1: class_names[0]}
        return adapted_model, adapted_class

    def adapted_model(x): return np.array(model_pred(
        x)) if pred_test[0] < 0.5 else 1 - np.array(model_pred(x))
    adapted_class = {0: class_names[0], 1: class_names[1]} if pred_test[0] < 0.5 else {0: class_names[1],
                                                                                       1: class_names[0]}
    return adapted_model, adapted_class


def greedy_strategy(factual, cf, adapted_model, feature_names):
    modified_features = np.where(factual != cf)[0]

    features_data = []
    ordered_features = np.array([])
    current_factual = factual.copy()
    # Make a greedy search to find the best feature to change
    for _ in range(len(modified_features)):
        # Get features that were not added to the ordered_features
        check_features = np.array(
            [feat for feat in modified_features if feat not in ordered_features])

        # Create modified factuals with counterfactual values in the indexes of check_features
        modified_factuals = [current_factual.copy()
                             for _ in range(len(check_features))]
        for check_feature, modified_factual in zip(check_features, modified_factuals):
            modified_factual[check_feature] = cf[check_feature]

        # Make predictions
        modified_factuals_pred = adapted_model(np.array(modified_factuals))

        best_feature = check_features[np.argmax(modified_factuals_pred)]
        best_pred = np.max(modified_factuals_pred)
        best_feat_name = feature_names[best_feature]
        best_factual_feature_value = factual[best_feature]
        best_cf_feature_value = cf[best_feature]

        current_factual = modified_factuals[np.argmax(modified_factuals_pred)]

        # Adjust text of modification values
        factual_value = _convert_legendvalue(best_factual_feature_value)
        counterfactual_value = _convert_legendvalue(best_cf_feature_value)

        features_data.append({
            'score': best_pred,
            'name': best_feat_name,

            'factual': factual_value,
            'counterfactual': counterfactual_value,
        })
        ordered_features = np.append(ordered_features, best_feature)

    return features_data


def calc_adapted_instance(instance, instance2, _indices):
    return np.asarray([instance[i] if i not in _indices else instance2[i] for i in range(len(instance))])


def _bitfield2d(n):
    all_indices = range(2 ** len(n))
    return np.array([
        [
            (i >> bit) & 1
            for bit in range(len(n))
        ]
        for i in all_indices
    ])


def _get_subcoalitions(indices_feature_diff):
    # return all subcoalitions
    return [
        subset 
        for L in range(0, len(indices_feature_diff) + 1)
        for subset in combinations(indices_feature_diff, L)
    ]

def _array_key(arr: np.ndarray):
    """Convert a numpy array to a reconstructable, but hashable object"""
    return (arr.dtype.str, arr.shape, arr.tobytes())


def get_model_prediction(changed_instance, model, class_of_interest_index):
    """Wrapper function to get the model prediction of an adapted instance with caching.
    
    This function converts numpy array inputs to hashable objects to enable LRU caching
    of model predictions, which significantly improves performance when computing Shapley values.
    
    Note: The model object itself must be hashable for caching to work. If your model is not
    hashable (e.g., most scikit-learn models, TensorFlow models, PyTorch models), you should
    wrap it in a hashable callable or use a different caching strategy.
    
    Args:
        changed_instance: numpy array representing the instance to predict
        model: callable that takes a list of instances and returns predictions (must be hashable).
              Expected to return an array-like object that can be indexed by class_of_interest_index.
        class_of_interest_index: index of the class to extract from predictions
        
    Returns:
        The prediction value for the specified class
    """
    return _get_model_prediction_cached(
        _array_key(changed_instance),
        model,
        class_of_interest_index
    )


@lru_cache(maxsize=None)
def _get_model_prediction_cached(changed_instance_key, model, class_of_interest_index):
    """Internal cached function for model predictions.
    
    This function is decorated with @lru_cache to cache predictions based on the input
    parameters. The cache uses the hash of all parameters, including the model object.
    
    IMPORTANT: This function requires the model parameter to be hashable. Most ML model
    objects (scikit-learn, TensorFlow, PyTorch) are NOT hashable by default and will
    raise a TypeError. For those cases, wrap the model in a hashable callable or
    implement a custom caching strategy.
    
    Args:
        changed_instance_key: tuple of (dtype, shape, bytes) representing the numpy array
        model: callable prediction function (must be hashable for caching to work).
               Expected to return an array-like object that can be indexed by class_of_interest_index.
        class_of_interest_index: index of the class to extract from predictions
        
    Returns:
        The prediction value for the specified class
    """
    changed_instance = np.frombuffer(changed_instance_key[2], dtype=changed_instance_key[0]).reshape(changed_instance_key[1])
    return model([changed_instance])[class_of_interest_index]


def calc_contribution_graph(
    instance1: [], 
    instance2: [], 
    model, 
    indices_feature_diff,
    class_of_interest_index=0,
    sample_kwargs=None
    ) -> {}:
    """
    Changes all possible combinations of features of instance1 and calculates the prediction value.
    The features to be changed are given by index by :arg indices_feature_diff: and are changed to the values as they
    appear on these positions in :arg instance2:

    :param instance1: The instance to be changed on certain features
    :param instance2: The instance whose values the features will be changed to
    :param model: The model, scoring the instance (must contain a predict_proba() method)
    :param indices_feature_diff: The indices of the features whose value in :arg instance1: need to be changed to the
    values as they appear in :arg instance2: at the same index
    :param class_of_interest_index: predict_proba() predicts an outcome for each class. Pass the index of the class of
    relevance here

    :return: Dict: dictionary whose keys are the subset of features (some combination out of :arg indices_feature_diff:)
    and whose values are the prediction value of :arg instance1:, but with adapted features on the indices of said
    subset
    """
    contribution_graph = {}
    if sample_kwargs is None:
        # Generate all subcoalitions
        subcoalitions = _get_subcoalitions(indices_feature_diff)
    else:
        # Sample subcoalitions
        if "n" not in sample_kwargs:
            raise ValueError("sample_kwargs must contain 'n' key specifying the sample size")
        sample_size = sample_kwargs['n']
        ensure_outer_edges = sample_kwargs.get("ensure_outer_edges", True)
        subcoalitions = get_subcoalitions_sample(indices_feature_diff, sample_size=sample_size, ensure_outer_edges=ensure_outer_edges)
        logger.debug("Sampled {} unique coalitions".format(len(subcoalitions)))
    for subcoalition in subcoalitions:
        changed_instance = calc_adapted_instance(instance1, instance2, subcoalition)
        prediction = get_model_prediction(changed_instance, model, class_of_interest_index)
        contribution_graph[subcoalition] = prediction
    return contribution_graph

def get_shapley_feature_contribution(contribution_graph: dict, feature_index: int):
    """
    Calculate the Shapley contribution of a single feature.
    
    When sampling is used to build the contribution_graph, some coalitions may be missing.
    In such cases, the function skips comparisons where a coalition S or S∪{i} is not present,
    which can introduce bias in the Shapley value estimation. A warning is logged when more
    than 10% of potential comparisons are skipped due to missing coalitions.
    
    Args:
        contribution_graph: Dictionary mapping coalitions (tuples of feature indices) to predictions
        feature_index: Index of the feature to calculate Shapley contribution for
        
    Returns:
        float: Shapley contribution value for the feature
    """
    # number of players (maximum coalition size in the graph)
    n = max((len(S) for S in contribution_graph), default=0)

    # precompute Shapley weights by coalition size
    weights = {
        k: factorial(k) * factorial(n - k - 1) / factorial(n)
        for k in range(n)
    }

    contribution = 0.0
    # Track comparisons to detect incomplete sampling
    attempted_comparisons = 0
    skipped_comparisons = 0
    
    for S, v_S in contribution_graph.items():
        if feature_index in S: continue
        S_with_i = tuple(sorted((*S, feature_index)))
        attempted_comparisons += 1
        v_S_i = contribution_graph.get(S_with_i)
        if v_S_i is None:
            skipped_comparisons += 1
            continue
        k = len(S)
        contribution += (v_S_i - v_S) * weights[k]

    # Warn if a significant proportion of comparisons were skipped
    if attempted_comparisons > 0 and skipped_comparisons / attempted_comparisons > 0.1:
        logger.warning(
            f"Feature {feature_index}: Skipped {skipped_comparisons}/{attempted_comparisons} "
            f"({100 * skipped_comparisons / attempted_comparisons:.1f}%) coalition comparisons due to sampling. "
            "This may introduce bias in the Shapley value estimate."
        )

    return contribution


def calc_shapley_values_between(
    instance1, 
    instance2, 
    model, 
    class_of_interest_index=0,
    sample_kwargs = None,
    return_contribution_graph = False,
    ):
    """
    Calculates shapley values between some instance and another instance, where the differences
    :param instance1: 1-dimensional array: The first instance (factual)
    :param instance2: 1-dimensional array: The second instance (counterfactual)
    :param model: the modal generating predictions. Must have a predict_proba() method
    :param class_of_interest_index: The index of the target class of interest. Generally equals 0
    :return: feature_contributions, indices
    """
    # where counterfactual is different from factual
    indices_feature_diff = np.where(instance1 != instance2)[-1]
    contribution_graph = calc_contribution_graph(
        instance1=instance1, 
        instance2=instance2, 
        model=model, 
        indices_feature_diff=indices_feature_diff,
        class_of_interest_index=class_of_interest_index,
        sample_kwargs=sample_kwargs
    )

    feature_contributions = []
    for feature in indices_feature_diff:
        contr = get_shapley_feature_contribution(contribution_graph, feature)
        feature_contributions.append(contr)
    # sort based on their contribution, large to small
    feature_contributions, feature_idx_sorted = zip(
        *sorted(zip(feature_contributions, indices_feature_diff))[::-1]
        )
    if return_contribution_graph: 
        return feature_contributions, feature_idx_sorted, contribution_graph
    return feature_contributions, feature_idx_sorted


def _format_e(x):
    return '{:.2e}'.format(x).replace('e+0', 'E').replace('e-0', 'E-').replace('e+', 'E').replace('e-', 'E-')


def _convert_legendvalue(x):
    # Verify if x is a number
    try:
        xn = float(x)
        # Verify if number has decimal places
        if xn % 1 == 0:
            # If number > 9999, convert to scientific notation
            if xn > 9999:
                return _format_e(xn)
            else:
                return x
        else:
            # If decimal places take more than 5 characters
            if abs(round(xn, 5)) <= 0.00001:
                return _format_e(xn)
            else:
                return round(xn, 5)
    except ValueError:
        return x
