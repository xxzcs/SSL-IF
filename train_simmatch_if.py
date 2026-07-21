#!/usr/bin/env python3
# 独立启动器: 注册新算法 'simmatch_if' 后转交 train.py。不修改任何原始文件。
# 用法: python train_simmatch_if.py --c config/usb_cv/simmatch_if/xxx.yaml
import os
import runpy

from semilearn.algorithms.simmatch_if import SimMatchIF
from semilearn.core.utils import ALGORITHMS
ALGORITHMS['simmatch_if'] = SimMatchIF
try:
    import semilearn.algorithms as _algs
    if hasattr(_algs, 'name2alg') and isinstance(_algs.name2alg, dict):
        _algs.name2alg['simmatch_if'] = SimMatchIF
except Exception as _e:  # pragma: no cover
    print('[train_simmatch_if] name2alg patch skipped:', _e)

_here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_here, 'train.py'), run_name='__main__')
