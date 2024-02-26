# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:03:22 2024

@author: LP275843
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.EigenmodeRayleighReal import seekEigenMode, getEigenMode


""" Function calculating the eigenmode and displaying the natural frequencies
of the problem
"""
NFi = 50

omegaMin = 10**(-5)
omegaMax = 60

Resol = 150

model = "Lagrange"
# model = "Yeh"

beam = CaseStudy()

print(beam)

lst = seekEigenMode(omegaMin, omegaMax, Resol, beam, model, NFi, False)

print(lst)

def w(x,i):
    return getEigenMode(lst[i], beam, model, NFi, False)(x)

ResolTrac = 100

X = np.linspace(0, 3, ResolTrac)
Y = np.zeros((ResolTrac, len(lst)))


###############################################################################
## Plot figure
###############################################################################



mpl.rcParams["text.usetex"]=True
mpl.rcParams["font.family"]="Helvetica"

fig, ax = plt.subplots()

ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()


ax.set_xlabel(r'Dimensionless axial coordinate, $x$')
ax.set_ylabel(r'Normalized free vibration mode, $w_0^{(j)}/||w_0^{(j)}||_{\infty}$')

ax.set_box_aspect(1/((1+np.sqrt(5))/2))

for i in range(len(lst)):
    for x in range (ResolTrac):
        Y[x,i] = w(X[x],i)
    Y[:,i] = (Y[:,i]/np.max(np.abs(Y[:,i])))*np.sign(Y[ResolTrac//2,i])
    if i == 0:
        param = 'k-'
    if i == 1:
        param = 'r--'
    if i == 2:
        param = 'b:'
    if i == 3:
        param = 'g-.'
    ax.plot(X, Y[:,i], param)

ax.set_xlim(left = 0, right = 3)
ax.set_ylim(bottom = -1.1, top = 1.3)
ax.text(0.40, 1.1, r'Blade')
ax.text(1.85, 1.1, r'Cylinder')
ax.plot([1, 1], [-1.1, 1.3], 'k--')
fig.savefig('Figure\\NaturalFqz.pdf', format="pdf", dpi=1200, bbox_inches='tight')
plt.show()

ax.clear()