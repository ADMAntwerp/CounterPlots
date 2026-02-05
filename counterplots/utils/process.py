from itertools import combinations
from functools import lru_cache
import numpy as np
from math import factorial


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


def _get_subcoalitions_sample(indices_feature_diff, sample_size):
    """Sample a subset of subcoalitions, ensuring for each subcoalition, 
    there is at least one other subcoaltion that differs by a single feature.
    
    Attention:
        These subcoalitions are NOT guaranteed to be unique.
        It is possible for the same coalition to be sampled for different feature indices.
        E.g. (A, B, C) can be sampled to measure (A, B) - (A, B, C), and again to measure (B, C) - (A, B, C)

        The focus of this function is to guarantee each sample has at least one parent or child so that it is useful
        for Shapley value calculation. Whether these parents or children are duplicates is of less concern.

    Raises:
        AssertionError: If the amount of requested samples exceeds the total amount of combinations. Just use exact calculation then.
    """
    assert sample_size < 2**len(indices_feature_diff), f"You are taking {sample_size} samples, but there are fewer possible combinations (2^{len(indices_feature_diff)} = {2**len(indices_feature_diff)})."
    n_feats = len(indices_feature_diff)
    n_samples_per_feat = int(np.ceil(sample_size / n_feats))
    indices_feature_diff = np.asarray(indices_feature_diff)
    subcoalitions = []
    subcoalitions.extend([tuple(), tuple(indices_feature_diff)])  # ensure at least the factual and cf are in here
    for feat in indices_feature_diff:                                       # O(k)
        # features excluding the current one
        subc_wo_feat = indices_feature_diff[indices_feature_diff != feat]   # O(k)
        
        m = len(subc_wo_feat)
        max_masks = 2 ** m
        n = min(n_samples_per_feat, max_masks)
        masks = np.random.choice(max_masks, n, replace=False)           
        bitmasks = ((masks[:, None] >> np.arange(m)) & 1).astype(bool)
        for mask in bitmasks:                                               # O(n_samples_per_feat) = O(N) / k
            subc1 = subc_wo_feat[mask]
            # Create version WITH feat and sort (sorting is important so the cache recognizes same input)
            subc2 = np.insert(subc1, np.searchsorted(subc1, feat), feat)    # O(k)
            subcoalitions.append(tuple(subc1))
            subcoalitions.append(tuple(subc2))
    
    return subcoalitions

def _array_key(arr: np.ndarray):
    """Convert a numpy array to a reconstructable, but hashable object"""
    return (arr.dtype.str, arr.shape, arr.tobytes())


def get_model_prediction(changed_instance, model, class_of_interest_index):
    """Wrapper function to get the model prediction of an adapted instance, where the input instances
    are first converted to reconstructable, but hashable (and therefore cachable) objects"""
    return _get_model_prediction_cached(
        _array_key(changed_instance),
        model,
        class_of_interest_index
    )


@lru_cache(maxsize=None)
def _get_model_prediction_cached(changed_instance_key, model, class_of_interest_index):
    """Reconstruct numpy arrays from their key and run model inference, or fetch it from cache"""
    changed_instance = np.frombuffer(changed_instance_key[2], dtype=changed_instance_key[0]).reshape(changed_instance_key[1])
    return model([changed_instance])[class_of_interest_index]


def calc_contribution_graph(
    instance1: [], 
    instance2: [], 
    model, 
    indices_feature_diff,
    class_of_interest_index=0,
    sample_size=None
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
    if sample_size is None:
        # Generate all subcoalitions
        subcoalitions = _get_subcoalitions(indices_feature_diff)
    else:
        # Sample subcoalitions
        subcoalitions = _get_subcoalitions_sample(indices_feature_diff, sample_size=sample_size)
    for subcoalition in subcoalitions:
        changed_instance = calc_adapted_instance(instance1, instance2, subcoalition)
        prediction = get_model_prediction(changed_instance, model, class_of_interest_index)
        contribution_graph[subcoalition] = prediction
    return contribution_graph


def get_shapley_feature_contribution(contribution_graph: None, feature_index: int):
    """Calculate the feature contributions of a single feature.
    
    Args:
        contribution_graph (dict): 
            A dictionary mapping each possible subcoalitoin of feature changes to their model score
            Subcoalitions are tuples containing feature indices, e.g. (1, 4, 9).
        feature_index (int): The index of the feature for which the Shapley contribution should be calulcated.
    """
    # amount of features/players
    n = max([len(node) for node in contribution_graph])
    ordered_nodes = [
        [node for node in contribution_graph if len(node) == i]
        for i in range(n+1)
        ]
    contributions_feature_i = []
    for i in range(len(ordered_nodes) - 1):
        row1 = ordered_nodes[i]
        row2 = [node for node in ordered_nodes[i + 1]
                if feature_index in node]
        size = len(row1[0])  # |S|, i.e. the size of the set S
        for node_w_feature in row2:
            remainder = [e for e in node_w_feature if e != feature_index]
            backlink = [node for node in row1 if all([i in node for i in remainder])][0]
            # contribution of adding feature i
            marginal_contr = contribution_graph[node_w_feature] - contribution_graph[backlink]
            marginal_contr_norm = marginal_contr * factorial(size) * factorial(n-size-1) / factorial(n)
            contributions_feature_i.append(marginal_contr_norm)
    # sum all contributions over all sets S
    return np.sum(contributions_feature_i)

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
    model, 
    class_of_interest_index=0,
    sample_size = None,
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
        sample_size=sample_size
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
