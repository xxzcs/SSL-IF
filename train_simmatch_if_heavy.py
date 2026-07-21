#!/usr/bin/env python3
# 启动器: 注册 simmatch_if_heavy(重型 Captum TracIn 版) 后转交 train.py。
import os, runpy
from semilearn.algorithms.simmatch_if.simmatch_if_heavy import SimMatchIFHeavy
from semilearn.core.utils import ALGORITHMS
ALGORITHMS['simmatch_if_heavy'] = SimMatchIFHeavy
try:
    import semilearn.algorithms as _algs
    if hasattr(_algs, 'name2alg') and isinstance(_algs.name2alg, dict):
        _algs.name2alg['simmatch_if_heavy'] = SimMatchIFHeavy
except Exception as _e:
    print('[train_simmatch_if_heavy] name2alg patch skipped:', _e)
_here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_here, 'train.py'), run_name='__main__')
