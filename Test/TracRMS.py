# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:20:53 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Integrand import Integrand

"""Script displaying the RMS as a function of x the dimensionless coordiante
"""

beam = CaseStudy()

beam.data['u'] = 0.32

Resol = 1000
NMatrix = 10
NFi = 50
omegaMin = 10**(-5)
omegaMax = 2.5*beam.data["u"]
model = "Lagrange"
# model = "Yeh"

array = Integrand(w, NMatrix, NFi, model, beam, omegaMin, omegaMax, Resol)[0]

X1 = np.linspace(0, 1, 40)
X2 = np.linspace(1, 1 + beam.data["h"], 40)

def RMSx(x):
    W = np.array([w(i, 0, x) for i in range(NMatrix)])
    RMS = np.sqrt((((beam.data["l"]/(beam.data["h"]*(beam.data["epsilon"]-1)))**2)/(2*np.pi))*np.einsum('i, ij, j->',W, array, W).real)
    return RMS

Y1 = 100*np.array([RMSx(x) for x in X1])
Y2 = 100*np.array([RMSx(x) for x in X2])



###############################################################################
## Plot figure
###############################################################################



mpl.rcParams["text.usetex"]=True
mpl.rcParams["font.family"]="Helvetica"

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, (1+np.sqrt(5))/2]})

ax1.grid(which = "major", linewidth = 1)
ax1.grid(which = "minor", linewidth = 0.2)
ax1.minorticks_on()

ax2.grid(which = "major", linewidth = 1)
ax2.grid(which = "minor", linewidth = 0.2)
ax2.minorticks_on()

ax2.get_yaxis().tick_right()

fig.text(0.5, 0.1, r'Dimensionless axial coordinate, $x$', ha='center')
ax1.set_ylabel(r'Dimensionless RMS of the displacement, $\sigma\left( \%\right)$')

ax1.set_box_aspect(((1+np.sqrt(5))/2))
ax2.set_box_aspect(1)

ax1.set_xlim(left = 0, right = 1)
ax1.set_ylim(bottom = -0.005, top = 0.135)

ax2.set_xlim(left = 1, right = 1+beam.data["h"])
ax2.set_ylim(bottom = -0.22, top = 0.7)

ax1.text(0.40, 0.12, r'Blade')
ax2.text(5.5, 0.6, r'Cylinder')

ax1.plot(X1, Y1, 'k-')
ax2.plot(X2, Y2, 'k-')

plt.subplots_adjust(wspace=0.05)

plt.show()

fig.savefig('Figure\\RMS_vs_x.pdf', format="pdf", dpi=1200, bbox_inches='tight')
ax1.clear()
ax2.clear()