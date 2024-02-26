# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 00:17:27 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("..")

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["text.usetex"]=True
mpl.rcParams["font.family"]="Helvetica"

from Main.AddCoef.FrictionFactor import FrictionFactor


""" Test displaying the friction coefficient as a function of the Reynold
"""


Resol = 100

Sk = np.logspace(np.log10(4000), 8, Resol)*2*np.pi
u = np.ones(Resol)

CT = FrictionFactor(Sk, u)[0]


###############################################################################
## Plot figure
###############################################################################



fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)

ax.plot(Sk/(2*np.pi), CT)

ax.set_box_aspect(1)

ax.set_xlabel(r'Reynolds, $Re$')
ax.set_ylabel(r'Tangent friction factor, $C_T$')

plt.show()