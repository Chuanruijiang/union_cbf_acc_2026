"""
In this file, we define the experiment setup for trajectory tracking tasks.
We are given an environment with:
static obstacles, 
a starting postion,
a goal position,
and also a reference trajectory (a collection of waypoints)
to follow.

We also provide a union of collection of CBFs:
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
import pydrake.symbolic as sym
from typing import Optional

import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ExperimentSetup:
    def __init__(self):
        self.waypoints = np.array([
            [4, 4],
            [8, 5],
            [12, 5],
            [16, 6],
            [16 + 2*np.sqrt(2), 6 + 2*np.sqrt(2)],
            [16 + 2*np.sqrt(2), 11 + 2*np.sqrt(2)],
            [18 + 2*np.sqrt(2), 15 + 2*np.sqrt(2)],
            [19 + 4*np.sqrt(2), 13 + 4*np.sqrt(2)],
            [23 + 4*np.sqrt(2), 11 + 4*np.sqrt(2)],
            [23 + 6*np.sqrt(2), 10 + 2*np.sqrt(2)],
            [23 + 6*np.sqrt(2), 8]
        ])
        
    def get_cbfs(self, x: np.ndarray, radius: float=3.0) -> np.ndarray:
        assert x.shape == (2,)
        return np.array([
            sym.Polynomial(radius**2 - (x[0] - each_point[0])**2 - (x[1] - each_point[1])**2)
            for each_point in self.waypoints
        ])
    
    
    def plot_setup(
            self,
            ax: Optional[matplotlib.axes.Axes]=None
        ):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        centers = self.waypoints
        radius = 3.0
        for each_center in centers:
            circle = patches.Circle(
                each_center,
                radius,
                facecolor='green',
                edgecolor=None,
                alpha=0.3
            )
            ax.add_patch(circle)
            ax.plot(
                each_center[0],
                each_center[1],
                markersize=7,
                marker='o',
                color=(0, 0.5, 0)
                )
        
        obstacle_1 = patches.Rectangle(
            xy=(5, 10),
            width=10,
            height=20,
            facecolor='black',
            edgecolor=None,
            alpha=0.5
        )
        ax.add_patch(obstacle_1)

        obstacle_2 = patches.Rectangle(
            xy=(23, 0),
            width=4,
            height=13,
            facecolor='black',
            edgecolor=None,
            alpha=0.5
        )
        ax.add_patch(obstacle_2)


def main():
    env_setup = ExperimentSetup()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    env_setup.plot_setup(ax=ax)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 35)
    legend_elements = [
        plt.Line2D([0], [0],
            color='blue', lw=1.5, linestyle="--", label='Switching Policy 1'
            ),
        plt.Line2D([0], [0],
            color='orange', lw=2, linestyle="-", label='Switching Policy 2'
            ),
        plt.Line2D([0], [0],
            color=(0, 1, 0), lw=2, linestyle='-.', label='CBF boundary'
            ),
        patches.Patch(facecolor='black', edgecolor='black', alpha=0.5, label='Obstacles'),
        patches.Patch(facecolor='green', alpha=0.3, label='CBF regions')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10
        )

if __name__ == "__main__":
    main()