# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core.utils import ALGORITHMS
from .adamatch import AdaMatch
from .comatch import CoMatch
from .crmatch import CRMatch
from .dash import Dash
from .defixmatch import DeFixMatch
from .fixmatch import FixMatch, FixMatch_IF
from .flexmatch import FlexMatch
from .freematch import FreeMatch, FreeMatch_IF
from .fullysupervised import FullySupervised
from .meanteacher import MeanTeacher
from .mixmatch import MixMatch
from .pimodel import PiModel
from .pseudolabel import PseudoLabel
from .refixmatch import ReFixMatch
from .remixmatch import ReMixMatch
from .sequencematch import SequenceMatch
from .simmatch import SimMatch
from .simmatchv2 import SimMatchV2
from .softmatch import SoftMatch
from .uda import UDA
from .vat import VAT

name2alg = ALGORITHMS

def get_algorithm(args, net_builder, tb_log, logger):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm]( # name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.algorithm)}')



