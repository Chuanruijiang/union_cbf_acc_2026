import union_cbf_base.non_empty_subset as test
import pydrake.symbolic as sym
import numpy as np

def main():
    x = sym.MakeVectorContinuousVariable(2, "x")
    cbfs = np.array([
        (0.4)**2 - (x[0]-0.5)**2 - (x[1] - 0)**2,
        (0.4)**2 - (x[0]+0.5)**2 - (x[1] - 0)**2
    ])
    non_empty_region = test.get_non_empty_region(
        h = cbfs,    
        x = x
    )
    print(non_empty_region[0].activation_index)
    print(non_empty_region[1].activation_index)

if __name__ == "__main__":
    main()
