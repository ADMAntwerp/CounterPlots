from itertools import combinations

import numpy as np


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

        factual_value_type = type(best_factual_feature_value)
        counterfactual_value_type = type(best_cf_feature_value)

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


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def calc_adapted_instance(instance, instance2, _indices):
    return [instance[i] if i not in _indices else instance2[i] for i in range(len(instance))]


def generate_contribution_graph(instance1: [], instance2: [], model, indices_feature_diff,
                                class_of_interest_index=0) -> {}:
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
    # different lengths of subsets of features
    for L in range(0, len(indices_feature_diff) + 1):
        for subset in combinations(indices_feature_diff, L):
            changed_instance = calc_adapted_instance(
                instance1, instance2, subset)
            # you now iterate all possible feature combinations of length L
            prediction = model([changed_instance])[class_of_interest_index]
            contribution_graph[subset] = prediction
    return contribution_graph


def calc_shapley_values_between(instance1, instance2, model, class_of_interest_index=0):
    """
    Calculates shapley values between some instance and another instance, where the differences
    :param instance1: 1-dimensional array: The first instance (factual)
    :param instance2: 1-dimensional array: The second instance (counterfactual)
    :param model: the modal generating predictions. Must have a predict_proba() method
    :param class_of_interest_index: The index of the target class of interest. Generally equals 0
    :return: feature_contributions, indices
    """

    def get_shapley_feature_contribution(_shapley_graph: {}, feature_index: int):
        # amount of features/players
        n = max([len(node) for node in _shapley_graph])
        ordered_nodes = [[node for node in _shapley_graph if len(node) == i]
                         for i in range(n+1)]
        contributions_feature_i = []
        for i in range(len(ordered_nodes) - 1):
            row1 = ordered_nodes[i]
            row2 = [node for node in ordered_nodes[i + 1]
                    if feature_index in node]
            size = len(row1[0])  # |S|, i.e. the size of the set S
            for node_w_feature in row2:
                remainder = [e for e in node_w_feature if e != feature_index]
                backlink = [node for node in row1 if all(
                    [i in node for i in remainder])][0]
                # contribution of adding feature i
                marginal_contr = shapley_graph[node_w_feature] - \
                    shapley_graph[backlink]
                marginal_contr_norm = marginal_contr * \
                    factorial(size) * factorial(n-size-1) / factorial(n)
                contributions_feature_i.append(marginal_contr_norm)
        # sum all contributions over all sets S
        return np.sum(contributions_feature_i)

    # where counterfactual is different from factual
    indices_feature_diff = np.where(instance1 != instance2)[-1]
    shapley_graph = generate_contribution_graph(instance1, instance2, model, indices_feature_diff,
                                                class_of_interest_index)

    feature_contributions = []
    for feature in indices_feature_diff:
        contr = get_shapley_feature_contribution(shapley_graph, feature)
        feature_contributions.append(contr)
    feature_contributions, ind = zip(
        *sorted(zip(feature_contributions, indices_feature_diff)))
    return feature_contributions[::-1], ind[::-1]


def _format_e(x):
    return '{:.2e}'.format(x).replace('e+0', 'E').replace('e-0', 'E-').replace('e+', 'E').replace('e-', 'E-')


def _convert_legendvalue(x):
    # Verify if x is a number
    try:
        xn = float(x)
        # Verify if number has decimal places
        if xn % 1 == 0:
            # If number > 1000, convert to scientific notation
            if xn > 1000:
                return _format_e(xn)
            else:
                return x
        else:
            return _format_e(xn)
    except ValueError:
        return x
