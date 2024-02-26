# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:31:21 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Integrand import Integrand
from Test.BenchmarkRMS import RMSThesisMaud2024, RMSErcofact2023


""" Function displaying the RMS at the tip end as a function of the Reynolds
"""

BeamWidth = 100*10**(-3)
BeamThickness = [14.5*10**(-3), 12*10**-3, 10*10**-3]
zeta = [0.46887022835276243, 0.6233043865088962, 0.8185053508630611]

ReMin = 4000
ReMax = 400000

nbPointCalc = 100

Resol = 1000
NMatrix = 5

NFi = 50

model = "Lagrange"
# model = "Yeh"

Re = np.logspace(np.log10(ReMin), np.log10(ReMax), nbPointCalc)
Y = np.zeros((len(BeamThickness), nbPointCalc))
YBenchmark = np.zeros((len(BeamThickness), nbPointCalc))

for ep in range (len(BeamThickness)):

    beam = CaseStudy()
    
    print(beam)
    
    beam.data["S"] = BeamWidth * BeamThickness[ep]
    beam.data["I"] = BeamWidth * (BeamThickness[ep] ** 3) / 12
    
    beam.Adim()
    
    beam.data["zeta"] = zeta[ep]
    
    print(beam)

    uMin = 2 * np.pi * ReMin / beam.data['Sk']
    uMax = 2 * np.pi * ReMax / beam.data['Sk']
    
    X = np.logspace(np.log10(uMin), np.log10(uMax), nbPointCalc)

    for i in range (nbPointCalc):
        print("")
        print(str((100*i)//nbPointCalc)+" %")
        print("Re = "+str(X[i] * beam.data['Sk'] / (2 * np.pi)))
        beam.data["u"]=X[i]
    
        omegaMin = 10**(-5)
        omegaMax = 2.5*beam.data["u"]
            
        array = Integrand(w, NMatrix, NFi, model, beam, omegaMin, omegaMax, Resol)[0]
        
        Y[ep, i] = 100 * np.sqrt((((beam.data["l"]/(beam.data["h"]*(beam.data["epsilon"]-1)))**2)/(2*np.pi))*np.einsum('ij->', array).real)
        YBenchmark[ep, i] = 100 * RMSErcofact2023(beam, omegaMin, omegaMax, **beam.data)



###############################################################################
## Plot figure
###############################################################################



mpl.rcParams["text.usetex"]=True
mpl.rcParams["font.family"]="Helvetica"

fig, ax = plt.subplots()

ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()

ax.set_box_aspect(2/(1+np.sqrt(5)))

ax.set_xlim(left = ReMin, right = ReMax)
ax.set_ylim(bottom = 10**(-4), top = 10**2)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r'Reynolds number, $Re$')
ax.set_ylabel(r'Dimensionless RMS of the displacement, $\sigma\left( \%\right)$')

for i in range (len(BeamThickness)):
    if i == 0:
        param = 'k-'
    if i == 1:
        param = 'r-'
    if i == 2:
        param = 'b-'
    plt.plot(Re, Y[i,:], param)
    
for i in range (len(BeamThickness)):
    if i == 0:
        param = 'k--'
    if i == 1:
        param = 'r--'
    if i == 2:
        param = 'b--'
    plt.plot(Re, YBenchmark[i,:], param)
    
plt.plot([1.74*(10**5), 1.74*(10**5)], [10**(-4), 10**2], 'k:')
plt.plot([1.18*(10**5), 1.18*(10**5)], [10**(-4), 10**2], 'r:')
plt.plot([7.70*(10**4), 7.70*(10**4)], [10**(-4), 10**2], 'b:')

fig.savefig('Figure\\RMS_vs_Re_LogLog.pdf', format="pdf", dpi=1200, bbox_inches='tight')
plt.show()