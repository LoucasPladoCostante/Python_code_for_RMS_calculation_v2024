# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:44:08 2024

@author: LP275843
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("..")

from Main.AddCoef.AddCoefViscLagr import FiViscLagr
from Main.AddCoef.AddCoefPerfLagr import FiPerfLagr

""" Function displaying $F^{(i)}$ as a function of the aspect ratio
"""

Resol = 50

epsilon = 1.14
Sk = 250
omega = 1

X = np.logspace(np.log10(0.5), np.log10(200), Resol)


Y1PerfLagr = np.zeros(Resol, complex)
Y1ViscLagr = np.zeros(Resol, complex)

Y2PerfLagr = np.zeros(Resol, complex)
Y2ViscLagr = np.zeros(Resol, complex)

for i in range (Resol):
    sys.stdout.write('\r'+str(100*i//Resol)+" %")
    f1PerfLagr, f2PerfLagr = FiPerfLagr(50, omega, X[i], epsilon, Sk, False)
    sys.stdout.write('\r'+str(100*(i+0.5)//Resol)+" %")
    f1ViscLagr, f2ViscLagr = FiViscLagr(50, omega, X[i], epsilon, Sk, False)
    Y1PerfLagr[i] = f1PerfLagr
    Y1ViscLagr[i] = f1ViscLagr
    Y2PerfLagr[i] = f2PerfLagr
    Y2ViscLagr[i] = f2ViscLagr

sys.stdout.write('\r end')


###############################################################################
## Plot figure
###############################################################################


mpl.rcParams["text.usetex"]=True
mpl.rcParams["font.family"]="Helvetica"
fig, ax = plt.subplots()


ax.set_xlim(left = 0.5, right = 200)
ax.set_xscale('log')
ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()

ax.plot(X, Y1ViscLagr.real, 'k-',label = r'Lagrange Model $Sk = 250$')
ax.plot(X, Y1PerfLagr.real, 'k--',label = r'Lagrange Model $Sk = \infty$')
ax.set_box_aspect(1)

ax.set_xlabel(r'Aspect ratio, $l$')
ax.set_ylabel(r'Added coefficient function, $\Re \left\{F^{(1)}\right\}$')

fig.savefig('Figure\\ReF1.pdf', format="pdf", dpi=1200, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots()

ax.set_xlim(left = 0.5, right = 200)
ax.set_xscale('log')
ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()

ax.plot(X, -Y1ViscLagr.imag, 'k-',label = r'Lagrange Model $Sk = 250$')
ax.plot(X, -Y1PerfLagr.imag, 'k--',label = r'Lagrange Model $Sk = \infty$')

ax.set_box_aspect(1)

ax.set_xlabel(r'Aspect ratio, $l$')
ax.set_ylabel(r'Added coefficient function, $-\Im \left\{F^{(1)}\right\}$')

fig.savefig('Figure\\ImF1.pdf', format="pdf", dpi=1200, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()

ax.set_xlim(left = 0.5, right = 200)
ax.set_xscale('log')
ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()

ax.plot(X, Y2ViscLagr.real, 'k-',label = r'Lagrange Model $Sk = 250$')
ax.plot(X, Y2PerfLagr.real, 'k--',label = r'Lagrange Model $Sk = \infty$')

ax.set_box_aspect(1)

ax.set_xlabel(r'Aspect ratio, $l$')
ax.set_ylabel(r'Added coefficient function, $\Re \left\{F^{(2)}\right\}$')

fig.savefig('Figure\\ReF2.pdf', format="pdf", dpi=1200, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()

ax.set_xlim(left = 0.5, right = 200)
ax.set_xscale('log')
ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()
ax.plot(X, -Y2ViscLagr.imag, 'k-',label = r'Lagrange Model $Sk = 250$')
ax.plot(X, -Y2PerfLagr.imag, 'k--',label = r'Lagrange Model $Sk = \infty$')

ax.set_box_aspect(1)

ax.set_xlabel(r'Aspect ratio, $l$')
ax.set_ylabel(r'Added coefficient function, $-\Im \left\{F^{(2)}\right\}$')

fig.savefig('Figure\\ImF2.pdf', format="pdf", dpi=1200, bbox_inches='tight')
plt.show()

ax.clear()

