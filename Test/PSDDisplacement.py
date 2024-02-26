# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:20:53 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Integrand import Integrandlog

"""
Function displaying the PSD of the amplitude of the two first mode
"""

beam = CaseStudy()

beam.data["u"] = 0.32

Resol = 1000
NMatrix = 5
NFi = 50
omegaMin = 10**(-3)
omegaMax = 2.5*beam.data["u"]
model = "Lagrange"
# model = "Yeh"

Omega = np.logspace(np.log10(omegaMin), np.log10(omegaMax), Resol)
array = Integrandlog(w, NMatrix, NFi, model, beam, omegaMin, omegaMax, Resol)



###############################################################################
## Plot figure
###############################################################################



mpl.rcParams["text.usetex"]=True
mpl.rcParams["font.family"]="Helvetica"

size = 1

fig, ax = plt.subplots()

ax.set_yscale("log")
ax.set_xscale("log")

ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()

ax.set_xlabel(r'Dimensionless angular frequency, $\omega$')
ax.set_ylabel(r'Dimensionless PSD, $<\bf{Q}|\bf{Q}>_{\it{j},\it{j}}$')

ax.set_box_aspect(2/(1+np.sqrt(5)))

ax.set_xlim(left = omegaMin, right = omegaMax)
ax.set_ylim(bottom = 10**(-2), top = 10**7)

plt.plot(Omega,array[:,0,0].real,'k-')
plt.plot(Omega,array[:,1,1].real,'r--')
plt.plot([beam.data["u"]/2, beam.data["u"]/2],[10**(-2), 10**7], 'k:' )
fig.savefig('Figure\\PSD_Displ.pdf', format="pdf", dpi=1200, bbox_inches='tight')
plt.show()

