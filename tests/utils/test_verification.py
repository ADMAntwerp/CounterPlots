import pytest
import numpy as np
from unittest.mock import Mock

from counterplots.utils.verification import verify_factual_counterfactual_shape, verify_factual_cf, \
    verify_feature_names, verify_model_prediction, verify_binary_class, verify_class_names, verify_shape


def test_verify_shape():
    array = np.array([1, 2, 3])
    verify_shape(array)  # No exception should be raised

    array_wrong_shape = np.array([[1, 2, 3]])  # Incorrect shape
    with pytest.raises(ValueError):
        verify_shape(array_wrong_shape)


def test_verify_factual_counterfactual_shape():
    factual = np.array([1, 2, 3])
    cf = np.array([4, 5, 6])
    verify_factual_counterfactual_shape(factual, cf)  # No exception should be raised

    cf_wrong_shape = np.array([[4, 5]])  # Incorrect shape
    with pytest.raises(ValueError):
        verify_factual_counterfactual_shape(factual, cf_wrong_shape)


def test_verify_factual_cf():
    factual = np.array([1, 2, 3])
    cf = np.array([4, 5, 6])
    verify_factual_cf(factual, cf)  # No exception should be raised

    factual_wrong_values = np.array([1, 2, 3, 4])  # Different number of features
    with pytest.raises(ValueError):
        verify_factual_cf(factual_wrong_values, cf)


def test_verify_feature_names():
    factual = np.array([1, 2, 3])
    feature_names = ['f1', 'f2', 'f3']
    verify_feature_names(factual, feature_names)  # No exception should be raised

    feature_names_wrong_length = ['f1', 'f2']  # Incorrect length
    with pytest.raises(ValueError):
        verify_feature_names(factual, feature_names_wrong_length)


def test_verify_model_prediction():
    factual = np.array([1, 2, 3])
    model_pred_correct = Mock()
    model_pred_correct.predict_proba = Mock(return_value=np.array([0]))
    verify_model_prediction(model_pred_correct, factual)  # No exception should be raised

    model_pred_incorrect = ''
    with pytest.raises(ValueError):
        verify_model_prediction(model_pred_incorrect, factual)


def test_verify_binary_class():
    factual = np.array([1, 2, 3])
    model_pred = lambda x: np.array([1]) if np.sum(x) > 5 else np.array([0])
    verify_binary_class(model_pred, factual)  # No exception should be raised

    model_pred_multiclass = lambda x: np.array([[0.1, 0.5, 0.4]])  # Multiclass prediction
    with pytest.raises(ValueError):
        verify_binary_class(model_pred_multiclass, factual)

    model_pred_other = lambda x: np.array([[[1]]])
    with pytest.raises(ValueError):
        verify_binary_class(model_pred_other, factual)


def test_verify_class_names():
    class_names = {0: 'Class 0', 1: 'Class 1'}
    verify_class_names(class_names)  # No exception should be raised

    class_names_incorrect_type = 'Class Names'  # Incorrect type
    with pytest.raises(ValueError):
        verify_class_names(class_names_incorrect_type)

