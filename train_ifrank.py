#!/usr/bin/env python3
# 独立启动器: 注册新算法 'simmatch_ifrank' 后, 原样转交给 train.py 运行。
# 不修改任何原始文件。用法同 train.py, 例如:
#   python train_ifrank.py --c config/usb_cv/simmatch_ifrank/xxx.yaml
import os
import runpy

# 1) 导入新算法模块 -> 触发 @ALGORITHMS.register('simmatch_ifrank')
from semilearn.algorithms.simmatch_ifrank import SimMatchIFRank
from semilearn.core.utils import ALGORITHMS
ALGORITHMS['simmatch_ifrank'] = SimMatchIFRank

# 2) 若 train.py 通过 name2alg 字典查算法, 也补丁进去 (不改原文件, 仅运行时注入)
try:
    import semilearn.algorithms as _algs
    if hasattr(_algs, 'name2alg') and isinstance(_algs.name2alg, dict):
        _algs.name2alg['simmatch_ifrank'] = SimMatchIFRank
except Exception as _e:  # pragma: no cover
    print('[train_ifrank] name2alg patch skipped:', _e)

# 3) 转交给原 train.py 的 __main__ (沿用当前 sys.argv)
_here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_here, 'train.py'), run_name='__main__')
