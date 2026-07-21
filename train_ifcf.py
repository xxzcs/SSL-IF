#!/usr/bin/env python3
# 独立启动器: 注册 freematch_ifcf / flexmatch_ifcf 后转交 train.py。不修改任何原始文件。
# 用法: python train_ifcf.py --c config/.../xxx.yaml   (config 里 algorithm: freematch_ifcf 或 flexmatch_ifcf)
import os
import runpy

from semilearn.algorithms.freematch.freematch_ifcf import FreeMatchIFcf
from semilearn.algorithms.flexmatch.flexmatch_ifcf import FlexMatchIFcf
from semilearn.algorithms.softmatch.softmatch_ifcf import SoftMatchIFcf
from semilearn.algorithms.adamatch.adamatch_ifcf import AdaMatchIFcf
from semilearn.algorithms.fixmatch.fixmatch_ifcf import FixMatchIFcf
from semilearn.algorithms.refixmatch.refixmatch_ifcf import ReFixMatchIFcf
from semilearn.algorithms.defixmatch.defixmatch_ifcf import DeFixMatchIFcf
from semilearn.core.utils import ALGORITHMS

ALGORITHMS['freematch_ifcf'] = FreeMatchIFcf
ALGORITHMS['flexmatch_ifcf'] = FlexMatchIFcf
ALGORITHMS['softmatch_ifcf'] = SoftMatchIFcf
ALGORITHMS['adamatch_ifcf'] = AdaMatchIFcf
ALGORITHMS['fixmatch_ifcf'] = FixMatchIFcf
ALGORITHMS['refixmatch_ifcf'] = ReFixMatchIFcf
ALGORITHMS['defixmatch_ifcf'] = DeFixMatchIFcf
try:
    import semilearn.algorithms as _algs
    if hasattr(_algs, 'name2alg') and isinstance(_algs.name2alg, dict):
        _algs.name2alg['freematch_ifcf'] = FreeMatchIFcf
        _algs.name2alg['flexmatch_ifcf'] = FlexMatchIFcf
        _algs.name2alg['softmatch_ifcf'] = SoftMatchIFcf
        _algs.name2alg['adamatch_ifcf'] = AdaMatchIFcf
        _algs.name2alg['fixmatch_ifcf'] = FixMatchIFcf
        _algs.name2alg['refixmatch_ifcf'] = ReFixMatchIFcf
        _algs.name2alg['defixmatch_ifcf'] = DeFixMatchIFcf
except Exception as _e:  # pragma: no cover
    print('[train_ifcf] name2alg patch skipped:', _e)

_here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_here, 'train.py'), run_name='__main__')
