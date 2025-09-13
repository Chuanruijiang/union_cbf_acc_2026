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

import numpy as np
import pydrake.symbolic as sym

import matplotlib.axes
import matplotlib.contour
import matplotlib.pyplot as plt

class ExperimentSetup:
    def __init__(self, x: np.ndarray):
        assert x.shape == (2,)
        self.waypoints = np.array([
            [4, 9],
            [8, 10],
            [12, 10],
            [16, 11],
            [16 + 2*np.sqrt(2), 11 + 2*np.sqrt(2)],
            [16 + 2*np.sqrt(2), 15 + 2*np.sqrt(2)],
            [17 + 2*np.sqrt(2), 18 + 2*np.sqrt(2)],
            [17 + 4*np.sqrt(2), 18 + 4*np.sqrt(2)],
            [21 + 4*np.sqrt(2), 18 + 4*np.sqrt(2)],
            [21 + 6*np.sqrt(2), 18 + 2*np.sqrt(2)],
            [22 + 6*np.sqrt(2), 18]
        ])
        self.cbfs = np.array([
            sym.Polynomial(4.0**2 - (x[0] - each_point[0])**2 - (x[1] - each_point[1])**2)
            for each_point in self.waypoints
        ])
        