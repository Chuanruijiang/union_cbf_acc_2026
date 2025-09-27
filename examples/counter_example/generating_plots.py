"""
In this script, we wanted to use a super level set of a polynomial
to cover a square region: 
-1 ≤ x1 ≤ 1, -1 ≤ x2 ≤ 1
We try different polynomial degrees and plot the results.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
from typing import Tuple, Optional, Union

import pydrake.symbolic as sym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from union_cbf_base.non_empty_subset import Subset
from union_cbf_base.union_cbf import UnionCbf
from union_cbf_base.plot import(
    plot_2D_function,
    plot_union_region
)

def main():
    # polynomial: 
    x = sym.MakeVectorContinuousVariable(2, "x")
    
    # plot range: 
    x0_range = (-1.25, 2.75)
    x1_range = (-1.5, 1.5)
    sampling_rate = 500
    fig, ax = plt.subplots(figsize=(5, 4))

    # plot the square region:
    square = patches.Rectangle(
        (-1, -1), 2, 2, facecolor='black', alpha=0.3
        )
    ax.add_patch(square)

    # plot the boundary of polynomial CBF:
    n = 6
    poly = sym.Polynomial(x[0]**n + x[1]**n - 2.30) 
    plot_2D_function(
        ax=ax,
        f=poly,
        x=x,
        x_range=x0_range,
        y_range=x1_range,
        sampling_rate=sampling_rate,
        with_contour=True,
        color='blue',
        contour_line_style='-',
        contour_line_width=1.5
    )

    # plot the 4 linear CBFs:
    cbfs = np.array([
        sym.Polynomial(x[0] - 1.05),
        sym.Polynomial(-x[0] - 1.05),
        sym.Polynomial(x[1] - 1.05),
        sym.Polynomial(-x[1] - 1.05),
    ])
    set_of_colors = ['red', 'orange', 'purple', 'green']
    for i in range(cbfs.shape[0]):
        plot_2D_function(
            ax=ax,
            f=cbfs[i],
            x=x,
            x_range=x0_range,
            y_range=x1_range,
            sampling_rate=sampling_rate,
            with_contour=True,
            color=set_of_colors[i],
            contour_line_style='--',
            contour_line_width=1.5
        )
    
    # plot the union region of the 4 linear CBFs:
    plot_union_region(
        ax=ax,
        p=cbfs,
        x=x,
        x_range=x0_range,
        y_range=x1_range,
        sampling_rate=sampling_rate,
        color='green',
        alpha=0.3
    )
    
    ax.set_xlim(x0_range)
    ax.set_ylim(x1_range)
    ax.set_xlabel('x1', fontsize=16)
    ax.set_ylabel('x2', fontsize=16)
    ax.set_title(
        'Union of CBFs vs. Single CBF', 
        fontsize=16
        )
    lengend_elements = [
        plt.Line2D(
            [0], [0], color='blue', linewidth=1.5, 
            label=r'$h(x)=0$'
            ),
        plt.Line2D(
            [0], [0], color='red', linestyle='--', linewidth=1.5, 
            label=r'$h_1(x)=0$'
            ),
        plt.Line2D(
            [0], [0], color='orange', linestyle='--', linewidth=1.5,
            label=r'$h_2(x)=0$'
            ),
        plt.Line2D(
            [0], [0], color='purple', linestyle='--', linewidth=1.5,
            label=r'$h_3(x)=0$'
            ),
        plt.Line2D(
            [0], [0], color='green', linestyle='--', linewidth=1.5,
            label=r'$h_4(x)=0$'
            ),
        patches.Patch(
            facecolor='black', edgecolor=None, alpha=0.3, 
            label=r'$\mathcal{X}_u$'
            ),
        patches.Patch(
            facecolor='green', edgecolor=None, alpha=0.3,
            label=r'$\mathcal{X}_c$'
            )
    ]

    ax.legend(handles=lengend_elements, fontsize=12, loc='right')



if __name__ == "__main__":
    main()
