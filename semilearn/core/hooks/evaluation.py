# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


BEST_METRIC_KEY_MAP = {
    'acc': 'eval/top-1-acc',
    'balanced_acc': 'eval/balanced-acc',
}


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            best_metric_name = getattr(algorithm, 'best_metric', 'acc')
            if best_metric_name not in BEST_METRIC_KEY_MAP:
                raise ValueError(f"Unsupported best_metric: {best_metric_name}")
            best_metric_key = BEST_METRIC_KEY_MAP[best_metric_name]
            current_best_metric = algorithm.log_dict[best_metric_key]

            # update best metrics
            if current_best_metric > algorithm.best_eval_acc:
                algorithm.best_eval_acc = current_best_metric
                algorithm.best_eval_metric_name = best_metric_name
                algorithm.best_it = algorithm.it
    
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it, 'eval/best_metric_name': algorithm.best_eval_metric_name}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            if algorithm.best_eval_metric_name == 'balanced_acc':
                results_dict['test/best_acc'] = test_dict['test/balanced-acc']
            else:
                results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict
        