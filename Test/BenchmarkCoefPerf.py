# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:45:22 2024

@author: LP275843
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("..")

mpl.rcParams.update(mpl.rcParamsDefault)

from Main.AddCoef.AddCoefPerfYeh import FiPerfYeh
from Main.AddCoef.AddCoefPerfLagr import FiPerfLagr

"""
Test comparing Lagrange and Yeh model for a perfect fluid as a function of the
aspect ratio of the cylinder
"""

Resol = 50

N = 50

epsilon = 1.1
Sk = -1
omega = -1

X = np.logspace(-0.5, 2, Resol)
Y1Lagr, Y2Lagr = [], []

for i in range (len(X)):
    sys.stdout.write('\r'+str(100*i//Resol)+" %")
    y1, y2 = FiPerfLagr(N, omega, X[i], epsilon, Sk, display = False)
    Y1Lagr.append(np.real(y1)), Y2Lagr.append(np.real(y2))

y1, y2 = FiPerfYeh(omega, -1, epsilon, Sk, display = False)
Y1Yeh, Y2Yeh = [y1]*Resol, [y2]*Resol


###############################################################################
## Plot figure
###############################################################################


plt.xscale("log")
plt.xlabel(r'Cylinder aspect ratio, $l$')
plt.ylabel(r'Modal self-added mass, $\Re \left\{F^{(1)}\right\}$')
plt.plot(X, Y1Lagr, 'b-',label = 'Lagrange Model')
plt.plot(X, Y1Yeh, 'b--',label = 'Yeh Model')
plt.show()

plt.xscale("log")
plt.xlabel(r'Cylinder aspect ratio, $l$')
plt.ylabel(r'Modal self-added Inertia, $\Re \left\{F^{(2)}\right\}$')
plt.plot(X, Y2Lagr, 'b-',label = 'Lagrange Model')
plt.plot(X, Y2Yeh, 'b--',label = 'Yeh Model')
plt.show()

FiPerfLagr(N, omega, 10, epsilon, Sk, display = True)
FiPerfYeh(omega, -1, epsilon, Sk, display = True)