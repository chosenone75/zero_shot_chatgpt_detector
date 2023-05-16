import random
import numpy as np
import os
import yaml

from zero_shot_chatgpt_detector import llm


def update_config(config, base_config='configs/eval.yaml'):
    # Get default config from yaml
    with open(os.path.join(os.path.dirname(__file__), base_config)) as f:
        default_config = yaml.safe_load(f)

    # Update default config with user config
    # Note that the config is a nested dictionary, so we need to update it recursively
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(default_config, config)


def simple_gpt_detector(
                 examples,
                 conf,
                 base_conf='configs/eval.yaml'):
    """
    Function to generate prompts using APE.
    Parameters:
        examples: to be detected texts.
        conf: config.
        base_conf: the path of default conf.yaml .
    Returns:
        An evaluation result.
    """

    conf = update_config(conf, base_conf)
    eval_conf = conf['evaluation']

    res = likelihood_evaluator(examples, eval_conf)

    return res


def likelihood_evaluator(examples, config):
    """
    For each prompt, evaluate the likelihood of the data (output) given the prompt.
    Parameters:
        prompts: A list of prompts.
        eval_template: The template for the evaluation queries.
        eval_data: The data to use for evaluation.
        config: The configuration dictionary.
    Returns:
        A LikelihoodEvaluationResult object.
    """
    queries = []
    output_indices = []
    # TODO: add support for prefix prompt
    for example in examples:
        queries.append(example)
        output_indices.append((0, len(example)))

    # Instantiate the LLM
    model = llm.model_from_config(config['model'])

    log_probs, _ = model.log_probs(queries, output_indices)

    res = LikelihoodEvaluationResult(examples, log_probs, 1)

    return res


class LikelihoodEvaluationResult():
    """
    A class for storing the results of a likelihood evaluation. Supports
    sorting prompts by various statistics of the likelihoods.
    """

    def __init__(self, prompts, log_probs, num_samples):
        self.prompts = prompts
        self.log_probs = log_probs
        self.prompt_log_probs = self._compute_avg_likelihood(
            prompts, log_probs, num_samples)

    def _compute_avg_likelihood(self, prompts, log_probs, num_samples):
        i = 0
        prompt_log_probs = []
        for prompt in prompts:
            prompt_log_probs.append([])
            lps = log_probs[i]
            prompt_log_probs[-1].append(sum(lps) / len(lps))
            i += 1
        return prompt_log_probs

    def _agg_likelihoods(self, method):
        """For each prompt, compute a statistic of the likelihoods (e.g., mean, median, etc.)"""
        if method == 'mean':
            return [-np.mean(lps) for lps in self.prompt_log_probs]
        elif method == 'median':
            return [np.median(lps) for lps in self.prompt_log_probs]
        elif method == 'std':
            return [np.std(lps) for lps in self.prompt_log_probs]
        elif method == 'max':
            return [np.max(lps) for lps in self.prompt_log_probs]
        elif method == 'min':
            return [-np.min(lps) for lps in self.prompt_log_probs]
        elif method == 'iqm':
            return [-np.mean(np.percentile(lps, [1, 10])) for lps in self.prompt_log_probs]
        else:
            raise ValueError(
                f'Unknown method {method} for aggregating likelihoods')

    def sorted(self, method='mean'):
        import math
        if method == 'default':
            scores = self._agg_likelihoods('mean')
        else:
            scores = self._agg_likelihoods(method)
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted([math.exp(score) for score in scores])
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_likelihoods('mean')
        else:
            scores = self._agg_likelihoods(method)
        return self.prompts, scores

    def top1_perplexity(self):
        prompts, scores = self.sorted()
        return scores[0]

    def __str__(self):
        s = ''
        prompts, scores = self.sorted()
        s += 'log(p): prompt\n'
        s += '----------------\n'
        for prompt, score in list(zip(prompts, scores))[:100]:
            s += f'{score:.2f}: {prompt}\n'
        return s
