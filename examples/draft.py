import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from dataclasses import dataclass
from typing_extensions import Self
from typing import List, Optional, Tuple
import numpy as np
import itertools

import pydrake.symbolic as sym
import pydrake.solvers as solvers

from compatible_clf_cbf.utils import (
    BinarySearchOptions,
    ContainmentLagrangianDegree,
    ContainmentLagrangian,
    elementary_symetric_polynomials,
    lie_derivative,
    lower_lie_derivatives,
    check_array_of_polynomials,
    get_polynomial_result,
    new_sos_polynomial,
    solve_with_id,
    is_sos,
)


def main():
    position = np.linspace(0, 1.85, 3)
    velocity = np.linspace(-1, 0, 3)
    position, velocity = np.meshgrid(position, velocity)
    initial_state_1 = np.stack([position, velocity], axis=-1).reshape(-1, 2)
    initial_state_2 = -initial_state_1

    

if __name__ == "__main__":
    main()