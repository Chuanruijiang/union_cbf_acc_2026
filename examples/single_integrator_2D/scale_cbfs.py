# """
# In this file, we define the experiment setup for union CBF
# verification comparison tasks.
# We define a union of 11 sparsely located circular CBFs, with
# each of them has the same radius. The whole setup is defined
# in a ExperimentSetup class, which has a method to return the
# CBFs as array of sym.Polynomials given the state of the
# system. 
# 
# This script only defines the experiment setup, and the
# verification comparison with be conducted in another script.
# 
# In order to check these circular CBFs, we can use the main()
# in this script to visualize the environment setup.
# """

import numpy as np
import pydrake.symbolic as sym
from typing import Optional

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ExperimentSetup:
    def __init__(self):
        self.centers = np.array([
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
        self.radius = 3.0
        
    def get_cbfs(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (2,)
        return np.array([
            sym.Polynomial(
                self.radius**2 - (x[0] - each_point[0])**2 - (x[1] - each_point[1])**2
                )
            for each_point in self.centers
        ])
    
    def plot_setup(
            self,
            ax: Optional[matplotlib.axes.Axes]=None
        ):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        centers = self.centers
        radius = self.radius
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
                markersize=3,
                marker='o',
                color=(0, 0.5, 0)
                )

def main():
    """
    This function shows how the environment setup looks like
    """
    env_setup = ExperimentSetup()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    env_setup.plot_setup(ax=ax)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 35)

if __name__ == "__main__":
    main()