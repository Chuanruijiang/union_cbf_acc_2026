import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from union_cbf_base.union_cbf import UnionCbf
from union_cbf_base.non_empty_subset import Subset
import union_cbf_base.utils as utils
import pydrake.symbolic as sym
import numpy as np
import pydrake.solvers as solvers
import pickle

def main():
    
    num = 7
    one_active_cbf = np.eye(num, dtype=int)
    two_active_cbf = np.eye(num, dtype=int)
    for i in range(num-1):
        two_active_cbf[i, i+1] = 1
    active_index_list = np.vstack((one_active_cbf, two_active_cbf[:-1, :]))
    print(f"The active index list is:\n{active_index_list}")
    assert active_index_list.shape[0] == num + (num - 1)




if __name__ == "__main__":
    main()
