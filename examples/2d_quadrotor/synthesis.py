import os
import sys
import os.path
import pickle

import numpy as np
import pydrake.symbolic as sym

from compatible_clf_union_cbf.utils import (
    compute_minimum_on_boundary,
    serialize_polynomial,
    deserialize_polynomial,
    BackoffScale,
)
from compatible_clf_union_cbf.clf import ClfSynthesis
from compatible_clf_union_cbf.union_cbf import UnionCbfSynthesisGivenClf
from dynamics import (
    system_dynamics,
    state_equation_constraint,
    original_to_extended_state_space,
)

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../.."))

def main():
    pi = np.pi
    x = sym.MakeVectorContinuousVariable(7, "x")
    x_set = sym.Variables(x)



if __name__ == "__main__":
    main()