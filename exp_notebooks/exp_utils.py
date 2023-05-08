import time
from nice import NICE
from counterplots import CreatePlot


class CounterPlotNice(NICE):

    def __init__(self, predict_fn, *args, **kwargs):
        super().__init__(predict_fn, *args, **kwargs)
        self.predict_fn = predict_fn

    def explain(self, X, feature_names=None, class_names=None, target_class='other'):
        explanations = super().explain(X, target_class)

        out_exp = []

        for i in range(len(X)):
            out_exp.append(CreatePlot(factual=X[i],
                                      cf=explanations[i],
                                      model_pred=self.predict_fn,
                                      feature_names=feature_names,
                                      class_names=class_names))

        return out_exp


class counterplot_explainer:

    def __init__(self, cf_generator, feature_names, class_names, model):
        self.cf_generator = cf_generator
        self.feature_names = feature_names
        self.class_names = class_names
        self.model = model

    def explain(self, factual):
        cf = self.cf_generator(factual)
        exp = CreatePlot(
            factual=factual,
            cf=cf,
            model_pred=self.model,
            feature_names=self.feature_names,
            class_names=self.class_names)

        results = exp
        results_cv = results.countershapley_values()
        r_fn = results_cv['feature_names']
        r_fv = results_cv['feature_values']
        results_per_idx = {self.feature_names.index(
            fn): fv for fn, fv in zip(r_fn, r_fv)}
        return results, results_per_idx


class shap_explainer:

    def __init__(self, explainer):
        self.explainer = explainer

    def explain(self, factual):
        exp = self.explainer(factual)
        results = exp
        results_per_idx = {idx: val for idx, val in enumerate(results[1])}

        return results, results_per_idx


class lime_explainer:

    def __init__(self, explainer, model):
        self.explainer = explainer
        self.model = model

    def explain(self, factual):
        exp = self.explainer.explain_instance(
            factual, self.model, num_features=len(factual))
        results = exp
        results_lv_idx = {l[0]: l[1] for l in results.as_map()[1]}
        results_per_idx = {idx: results_lv_idx[idx]
                           for idx in range(len(factual))}
        return results, results_per_idx


def generate_explanations(data, exp_gen):
    gen_times = []
    results = []
    results_per_idx = []
    for i in range(len(data)):
        start_time = time.time()
        result, result_per_idx = exp_gen(data[i])
        gen_times.append(time.time() - start_time)
        results.append(result)
        results_per_idx.append(result_per_idx)

    return results, results_per_idx, gen_times
