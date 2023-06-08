import numpy as np

# Verifications
def verify_shape(array):
    if len(np.array(array).shape) != 1:
        raise ValueError('Array must be 1-dimensional.')


def verify_factual_counterfactual_shape(factual, cf):
    for type, data in zip(['Factual', 'Counterfactual'], [factual, cf]):
        if len(data.shape) != 1:
            raise ValueError(f'{type} array must be 1-dimensional. Got {data.shape}.')


def verify_factual_cf(factual, cf):
    if np.array(factual).shape != np.array(cf).shape:
        raise ValueError('Factual and counterfactual must have the same shape.')


def verify_feature_names(factual, feature_names):
    if len(feature_names) != np.array(factual).shape[0]:
        raise ValueError('Feature names must have the same length as the number of features.')


def verify_model_prediction(model_pred, factual):
    try:
        model_pred(np.array([factual]))
    except Exception as e:
        raise ValueError('Model prediction function must be callable but got error: {}'.format(e))


def verify_binary_class(model_pred, factual):
    model_pred_shape = model_pred(np.array([factual])).shape
    if len(model_pred_shape) == 1:
        return
    elif len(model_pred_shape) == 2:
        if model_pred_shape[1] > 2:
            raise ValueError('Model must be binary.')
    else:
        raise ValueError('Model must be binary.')


def verify_class_names(class_names):
    if type(class_names) != dict or len(class_names) != 2:
        raise ValueError(
            'Class names must be a dictionary, where int(0) is the class for prediction 0 and '
            'int(1) is the class for prediction 1.')
