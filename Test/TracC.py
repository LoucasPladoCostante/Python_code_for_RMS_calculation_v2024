# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 00:17:27 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib as mpl
import scipy.special as spe
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

"""Comparison with the explicit formulation of the coolebrok formulation
"""

a = 2.51 / np.logspace(np.log10(4000), 8, Resol)
Cf = 1 / (((2 * spe.lambertw(np.log(10)/(2*a)))/np.log(10))**2)
CTCoolebrok = np.pi*Cf/4

###############################################################################
## Plot figure
###############################################################################

phi = (1+np.sqrt(5)) / 2

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(Sk/(2*np.pi), CT, 'k--')
ax.plot(Sk/(2*np.pi), CTCoolebrok, 'r')

ax.set_xlim(left = 1000, right = 100000000)
ax.set_ylim(top = np.max(CT), bottom = np.min(CT))

ax.set_box_aspect(1/phi)

ax.set_xlabel(r'Reynolds, $Re$')
ax.set_ylabel(r'Tangential friction factor, $C_T$')

fig.savefig('Figure\\TangentialFrictionFactor.pdf', format="pdf", dpi=1200, bbox_inches='tight')

plt.show()