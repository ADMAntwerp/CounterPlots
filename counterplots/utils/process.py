from itertools import combinations
from functools import lru_cache
import numpy as np
from multiprocessing import Pool, cpu_count
from math import factorial
from .approx import _get_subcoalitions_sample
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

MULTIPROCESS_MIN_SIZE = 6  # CFE dimension at which it starts making sense to parallellize

class AdaptedModel:
    def __init__(self, model_pred, col, invert):
        self.model_pred = model_pred
        self.col = col
        self.invert = invert

    def __call__(self, x):
        p = np.array(self.model_pred(x))
        if self.col is not None:
            p = p[:, self.col]
        return 1 - p if self.invert else p

def create_adapted_model_and_class(factual, model_pred, class_names):
    pred = np.array(model_pred(np.array([factual])))

    if pred.ndim == 2:
        col = 1 if pred.shape[1] == 2 else 0
        factual_prob = pred[0, col]
    else:
        col = None
        factual_prob = pred[0]

    invert = factual_prob >= 0.5

    adapted_model = AdaptedModel(
        model_pred=model_pred,
        col=col,
        invert=invert,
    )

    adapted_class = (
        {0: class_names[1], 1: class_names[0]}
        if invert
        else {0: class_names[0], 1: class_names[1]}
    )

    return adapted_model, adapted_class


def greedy_strategy(factual, cf, predict_fn, feature_names):
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
        modified_factuals_pred = predict_fn(np.array(modified_factuals))

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


def _get_all_subcoalitions(indices_feature_diff):
    # return all subcoalitions
    return [
        subset 
        for L in range(0, len(indices_feature_diff) + 1)
        for subset in combinations(indices_feature_diff, L)
    ]


def _generate_subcoalitions(indices_feature_diff, sample_kwargs):
    # Convert numpy array to list for compatibility
    if hasattr(indices_feature_diff, 'tolist'):
        indices_feature_diff = indices_feature_diff.tolist()
    
    if sample_kwargs is None:
        return _get_all_subcoalitions(indices_feature_diff)

    if "n" not in sample_kwargs:
        raise ValueError("sample_kwargs must contain key 'n' specifying the sample size")

    return _get_subcoalitions_sample(
        indices_feature_diff,
        sample_size=sample_kwargs["n"],
        ensure_outer_edges=sample_kwargs.get("ensure_outer_edges", True),
    )


def _array_key(arr: np.ndarray):
    """Convert a numpy array to a reconstructable, but hashable object"""
    return (arr.dtype.str, arr.shape, arr.tobytes())


def get_model_prediction(changed_instance, predict_fn, class_of_interest_index):
    """Wrapper function to get the model prediction of an adapted instance, where the input instances
    are first converted to reconstructable, but hashable (and therefore cachable) objects"""
    return _get_model_prediction_cached(
        _array_key(changed_instance),
        predict_fn,
        class_of_interest_index
    )


@lru_cache(maxsize=None)
def _get_model_prediction_cached(changed_instance_key, predict_fn, class_of_interest_index):
    """Reconstruct numpy arrays from their key and run model inference, or fetch it from cache"""
    changed_instance = np.frombuffer(changed_instance_key[2], dtype=changed_instance_key[0]).reshape(changed_instance_key[1])  # dim = (n_feat,)
    pred = predict_fn([changed_instance])[class_of_interest_index]
    return pred


def _evaluate_subcoalition(
    subcoalition,
    instance1,
    instance2,
    predict_fn,
    class_of_interest_index
    ):
    changed_instance = calc_adapted_instance(instance1, instance2, subcoalition)
    prediction = get_model_prediction(
        changed_instance, predict_fn, class_of_interest_index
    )
    return subcoalition, prediction


def _calc_contribution_graph_serial(
    subcoalitions,
    instance1,
    instance2,
    predict_fn,
    class_of_interest_index,
    ):
    contribution_graph = {}

    for subcoalition in subcoalitions:
        changed_instance = calc_adapted_instance(
            instance1, instance2, subcoalition
        )
        contribution_graph[subcoalition] = get_model_prediction(
            changed_instance,
            predict_fn,
            class_of_interest_index,
        )

    return contribution_graph


def _calc_contribution_graph_parallel(
    subcoalitions,
    instance1,
    instance2,
    predict_fn,
    class_of_interest_index,
    cpu_count_,
    ):
    with Pool(cpu_count_) as pool:
        results = pool.starmap(
            _evaluate_subcoalition,
            [
                (sc, instance1, instance2, predict_fn, class_of_interest_index)
                for sc in subcoalitions
            ],
        )

    return dict(results)


def calc_contribution_graph(
    instance1,
    instance2,
    predict_fn,
    class_of_interest_index=0,
    sample_kwargs=None,
    multiproc_kwargs=None
    ):
    """Calculate the contribution graph

    A contribution graph is a dictionary that maps each subcoalition of 
    feature changes to their corresponding score.
    The possible feature changes are constructed based on the differences between
    :param:`instance1` and :param:`instance2`.

    This is generated once, and subcoalition comparisons for the calculation
    of Shapley values is fetched from this dictionary.
    If you pass :param:`sample_kwargs`, the contribution graph is sampled,
    rather than calculated exactly.

    Args:
        instance1 (np.ndarray): The baseline instance. Usually the factual instance.
        instance2 (np.ndarray): The counterfactual instance.
        predict_fn (Callable): A function that takes an instance an produces a prediction score.
        
    """
    indices_feature_diff = np.where(instance1 != instance2)[-1]
    cf_size = len(indices_feature_diff)

    # --- multiprocessing defaults ---
    if multiproc_kwargs is None:
        multiproc_kwargs = {
            "min_size": MULTIPROCESS_MIN_SIZE,
            "cpu_count": cpu_count(),
        }

    subcoalitions = _generate_subcoalitions(
        indices_feature_diff,
        sample_kwargs,
    )

    if cf_size < multiproc_kwargs['min_size']:
        return _calc_contribution_graph_serial(
            subcoalitions=subcoalitions,
            instance1=instance1,
            instance2=instance2,
            predict_fn=predict_fn,
            class_of_interest_index=class_of_interest_index
        )
    else:
        return _calc_contribution_graph_parallel(
            subcoalitions=subcoalitions,
            instance1=instance1,
            instance2=instance2,
            predict_fn=predict_fn,
            class_of_interest_index=class_of_interest_index,
            cpu_count_=multiproc_kwargs['cpu_count']
        )


def get_shapley_feature_contribution(contribution_graph: dict, feature_index: int):
    """
    Calculate the Shapley contribution of a single feature.
    """
    # number of players
    n = max(len(S) for S in contribution_graph)

    # precompute Shapley weights by coalition size
    weights = {
        k: factorial(k) * factorial(n - k - 1) / factorial(n)
        for k in range(n)
    }

    contribution = 0.0
    for S, v_S in contribution_graph.items():
        if feature_index in S: continue
        S_with_i = tuple(sorted((*S, feature_index)))
        v_S_i = contribution_graph.get(S_with_i)
        if v_S_i is None: continue
        k = len(S)
        contribution += (v_S_i - v_S) * weights[k]

    return contribution


def calc_shapley_values_between(
    instance1, 
    instance2, 
    predict_fn, 
    class_of_interest_index=0,
    sample_kwargs = None,
    multiproc_kwargs=None
    ):
    """
    Calculates shapley values between some instance and another instance, where the differences
    :param instance1: 1-dimensional array: The first instance (factual)
    :param instance2: 1-dimensional array: The second instance (counterfactual)
    :param predict_fn: the modal generating predictions. Must have a predict_proba() method
    :param class_of_interest_index: The index of the target class of interest. Generally equals 0
    :return: feature_contributions, indices
    """
    # where counterfactual is different from factual
    indices_feature_diff = np.where(instance1 != instance2)[-1]
    contribution_graph = calc_contribution_graph(
        instance1=instance1, 
        instance2=instance2, 
        predict_fn=predict_fn, 
        class_of_interest_index=class_of_interest_index,
        sample_kwargs=sample_kwargs,
        multiproc_kwargs=multiproc_kwargs
    )

    feature_contributions = []
    for feature in indices_feature_diff:
        contr = get_shapley_feature_contribution(contribution_graph, feature)
        feature_contributions.append(contr)
    # sort based on their contribution, large to small
    feature_contributions, feature_idx_sorted = zip(
        *sorted(zip(feature_contributions, indices_feature_diff))[::-1]
        )
    return feature_contributions, feature_idx_sorted, contribution_graph


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
