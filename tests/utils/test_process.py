import unittest
import numpy as np
from counterplots.utils.process import create_adapted_model_and_class, greedy_strategy, factorial, \
    calc_adapted_instance, generate_contribution_graph, calc_shapley_values_between, _convert_legendvalue


class TestScript(unittest.TestCase):

    def setUp(self):
        def model(data):
            out = []
            for x in data:
                if x[0] == 0.1 and x[1] == 0.2 and x[2] == 0.3:
                    out.append(0.0)
                elif x[0] == 0.4 and x[1] == 0.5 and x[2] == 0.6:
                    out.append(1.0)
                elif x[0] == 0.1 and x[1] == 0.5 and x[2] == 0.6:
                    out.append(0.4)
                elif x[0] == 0.4 and x[1] == 0.2 and x[2] == 0.6:
                    out.append(0.2)
                elif x[0] == 0.4 and x[1] == 0.5 and x[2] == 0.3:
                    out.append(0.3)
                elif x[0] == 0.1 and x[1] == 0.2 and x[2] == 0.6:
                    out.append(0.25)
                elif x[0] == 0.4 and x[1] == 0.2 and x[2] == 0.3:
                    out.append(0.4)
                elif x[0] == 0.1 and x[1] == 0.5 and x[2] == 0.3:
                    out.append(0.44)
            return np.array(out)
        self.model = model
        self.factual = np.array([0.1, 0.2, 0.3])
        self.cf = np.array([0.4, 0.5, 0.6])


    def test_create_adapted_model_and_class(self):
        class_names = ['A', 'B']
        model_pred = self.model
        factual = self.factual

        adapted_model, adapted_class = create_adapted_model_and_class(factual, model_pred, class_names)

        self.assertTrue(callable(adapted_model))
        self.assertEqual(adapted_class[0], 'A')
        self.assertEqual(adapted_class[1], 'B')

    def test_create_adapted_model_and_class_B_A(self):
        class_names = ['A', 'B']
        model_pred = self.model
        factual = self.cf

        adapted_model, adapted_class = create_adapted_model_and_class(factual, model_pred, class_names)

        self.assertTrue(callable(adapted_model))
        self.assertEqual(adapted_class[0], 'B')
        self.assertEqual(adapted_class[1], 'A')

    def test_create_adapted_model_and_class_double_pred(self):
        class_names = ['A', 'B']
        model_pred = lambda data: [[1 - self.model([x])[0], self.model([x])[0]] for x in data]
        factual = self.factual

        adapted_model, adapted_class = create_adapted_model_and_class(factual, model_pred, class_names)

        self.assertTrue(callable(adapted_model))
        self.assertEqual(adapted_class[0], 'A')
        self.assertEqual(adapted_class[1], 'B')

    def test_create_adapted_model_and_class_double_pred_B_A(self):
        class_names = ['A', 'B']
        model_pred = lambda data: [[1 - self.model([x])[0], self.model([x])[0]] for x in data]
        factual = self.cf

        adapted_model, adapted_class = create_adapted_model_and_class(factual, model_pred, class_names)

        self.assertTrue(callable(adapted_model))
        self.assertEqual(adapted_class[0], 'B')
        self.assertEqual(adapted_class[1], 'A')

    def test_greedy_strategy(self):
        factual = self.factual
        cf = self.cf
        adapted_model = self.model
        feature_names = ['X', 'Y', 'Z']

        features_data = greedy_strategy(factual, cf, adapted_model, feature_names)

        self.assertEqual(features_data,
                         [{'score': 0.44, 'name': 'Y', 'factual': 0.2, 'counterfactual': 0.5},
                          {'score': 0.4, 'name': 'Z', 'factual': 0.3, 'counterfactual': 0.6},
                          {'score': 1.0, 'name': 'X', 'factual': 0.1, 'counterfactual': 0.4}])

    def test_factorial(self):
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)
        self.assertEqual(factorial(5), 120)

    def test_calc_adapted_instance(self):
        instance = self.factual
        instance2 = self.cf
        indices = [0, 2]

        adapted_instance = calc_adapted_instance(instance, instance2, indices)

        self.assertEqual(adapted_instance, [0.4, 0.2, 0.6])

    def test_generate_contribution_graph(self):
        instance1 = self.factual
        instance2 = self.cf
        model = lambda x: np.array([0.7])
        indices_feature_diff = [0, 2]
        class_of_interest_index = 0

        contribution_graph = generate_contribution_graph(instance1, instance2, model, indices_feature_diff, class_of_interest_index)

        self.assertEqual(len(contribution_graph), 4)
        self.assertIn((), contribution_graph)
        self.assertIn((0,), contribution_graph)
        self.assertIn((2,), contribution_graph)
        self.assertIn((0, 2), contribution_graph)

    def test_calc_shapley_values_between(self):
        instance1 = self.factual
        instance2 = self.cf
        model = self.model

        class_of_interest_index = 0

        feature_contributions, indices = calc_shapley_values_between(
            instance1, instance2, model, class_of_interest_index)

        self.assertEqual(feature_contributions, (0.42166666666666663, 0.30166666666666664, 0.2766666666666666))
        self.assertEqual(indices, (1, 0, 2))

    def test__convert_legendvalue(self):
        self.assertEqual(_convert_legendvalue(10000), '1.00E4')
        self.assertEqual(_convert_legendvalue(0.00001), '1.00E-5')
        self.assertEqual(_convert_legendvalue(0.123456), 0.12346)
        self.assertEqual(_convert_legendvalue('ABC'), 'ABC')
