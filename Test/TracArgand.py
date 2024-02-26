# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:47:29 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Matrix import MatrixGeneration
from Main.AddCoef.AddCoefAdim import AddCoefAdim

"""
function searching for the eigenvalue of the linear system, and seeking some
instabilities
"""

###############################################################################
## beam and fluid model
###############################################################################

beam = CaseStudy()

NFi = 50
model = 'Lagrange'

print(beam)

###############################################################################
## nb test func and prec eigenmode
###############################################################################

N = 4
eps = 10**(-8)

###############################################################################
## Guess eigenfunction
###############################################################################

guessfqz = [0.010563140609536396j, 0.5946571996153761j] #blade 1
# guessfqz = [0.009610746551616079j, 0.5414521664054922j] #blade 2
# guessfqz = [0.008775097571763417j, 0.49462510114477787j] #blade 3

###############################################################################
## Range of speed
###############################################################################

speedMin = -0.5
speedMax = 0.500001

Resol = 101
Speed = np.linspace(speedMin, speedMax, Resol)

fqz = np.zeros((Resol, len(guessfqz)), complex)

###############################################################################
## index on special speed value
###############################################################################

SpeedPoint = np.linspace(-0.5, 0.5, 21)
IndexSpeedPoint = [np.argmin(np.abs(Speed-u)) for u in SpeedPoint]

for u in range(Resol):
    
    beam.data["u"] = Speed[u]
    print(Speed[u])
    
    for g in range(len(guessfqz)):
        conv = False
        while not conv:
            
            Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta = MatrixGeneration(N, w, **beam.data)
            m0, c0, j0, d0, cu, fd, ku, au, du = AddCoefAdim(guessfqz[g].imag, beam, model, NFi, **beam.data)
            
            M = Ms + (m0 + beam.data["mcyl"]) * Mm + (j0 + beam.data["jcyl"]) * Mj
            C = Cs + (c0 + cu) * Cc + (d0 + du) * Cd + au * Ca
            K = Ks + ((beam.data["mF"]-beam.data["mcyl"]) * beam.data["g"] + fd) * Kf + ku * Kk
            
            MCK = np.zeros((2*N,2*N), complex)
            
            invMK = -np.dot(np.linalg.inv(M),K)
            invMC = -np.dot(np.linalg.inv(M),C)
            
            for i in range(N):
                MCK[i,N+i] = 1
                for j in range(N):
                    MCK[N+i,j] = invMK[i,j]
                    MCK[N+i,N+j] = invMC[i,j]
                    
            eigVal = np.linalg.eig(MCK)[0]
            
            index = 0
            
            for i in range(2*N):
                if np.abs(guessfqz[g]-eigVal[index])>np.abs(guessfqz[g]-eigVal[i]):
                    index = i
                    
            if np.abs(guessfqz[g]-eigVal[index])<eps:
                conv = True
            guessfqz[g] = eigVal[index]
            
        fqz[u,g] = guessfqz[g]
    
Point = np.array([fqz[i] for i in IndexSpeedPoint])



###############################################################################
###############################################################################
## Plot figure
###############################################################################
###############################################################################



mpl.rcParams["text.usetex"]=True
mpl.rcParams["font.family"]="Helvetica"

###############################################################################
## Search for point loss stability
###############################################################################

indexLambda1 = np.argmin(np.abs(fqz[:,0].real))
indexLambda2 = np.argmin(np.abs(fqz[:,1].real))


###############################################################################
## Real part
###############################################################################

fig, ax = plt.subplots()

ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()

ax.set_box_aspect(1)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

ax.set_xlim(left = Speed[0]*beam.data["Sk"]/(2*np.pi), right = Speed[-1]*beam.data["Sk"]/(2*np.pi))
ax.set_xlabel(r'Reynolds number, $Re$')
ax.set_ylabel(r'sgn$\left(\Re\left\{\lambda^{(j)}\right\}\right)$')

plt.plot((Speed[:(indexLambda1)]*beam.data["Sk"]/(2*np.pi)), np.sign(np.array(fqz[:(indexLambda1),0]).real), "k--", linewidth = 2)

plt.plot([(Speed[indexLambda1]*beam.data["Sk"]/(2*np.pi)), (Speed[indexLambda1]*beam.data["Sk"]/(2*np.pi))],\
         [np.sign(fqz[indexLambda1-1,0].real), np.sign(fqz[indexLambda1+1,0].real)], "k:", linewidth = 2)
    
plt.plot((Speed[(indexLambda1+1):]*beam.data["Sk"]/(2*np.pi)), np.sign(np.array(fqz[(indexLambda1+1):,0]).real), "k-", linewidth = 2)




plt.plot((Speed[:(indexLambda2)]*beam.data["Sk"]/(2*np.pi)), np.sign(np.array(fqz[:(indexLambda2),1]).real), "r-", linewidth = 2)

plt.plot([ (Speed[indexLambda2]*beam.data["Sk"]/(2*np.pi)), (Speed[indexLambda2]*beam.data["Sk"]/(2*np.pi))],\
         [np.sign(fqz[indexLambda2-1,1].real), np.sign(fqz[indexLambda2+1,1].real)], "r:", linewidth = 2)
    
plt.plot((Speed[(indexLambda2+1):]*beam.data["Sk"]/(2*np.pi)), np.sign(np.array(fqz[(indexLambda2+1):,1]).real), "r--", linewidth = 2)

fig.savefig('Figure\\Argand_set_1_Real.pdf', format="pdf", dpi=1200, bbox_inches='tight')

plt.show()

###############################################################################
## Imaginary part
###############################################################################

fig, ax = plt.subplots()

ax.grid(which = "major", linewidth = 1)
ax.grid(which = "minor", linewidth = 0.2)
ax.minorticks_on()

ax.set_box_aspect(1)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

ax.set_xlim(left = Speed[0]*beam.data["Sk"]/(2*np.pi), right = Speed[-1]*beam.data["Sk"]/(2*np.pi))
ax.set_xlabel(r'Reynolds number, $Re$')
ax.set_ylabel(r'$\Im\left\{\lambda^{(j)}\right\}/\omega_0^{(j)}$')

plt.plot((Speed[:indexLambda1]*beam.data["Sk"]/(2*np.pi)), np.array(fqz[:indexLambda1,0]).imag/guessfqz[0].imag, "k--", linewidth = 2)
    
plt.plot((Speed[indexLambda1:]*beam.data["Sk"]/(2*np.pi)), np.array(fqz[indexLambda1:,0]).imag/guessfqz[0].imag, "k-", linewidth = 2)


plt.plot((Speed[:indexLambda2]*beam.data["Sk"]/(2*np.pi)), np.array(fqz[:indexLambda2,1]).imag/guessfqz[1].imag, "r-", linewidth = 2)
    
plt.plot((Speed[indexLambda2:]*beam.data["Sk"]/(2*np.pi)), np.array(fqz[indexLambda2:,1]).imag/guessfqz[1].imag, "r--", linewidth = 2)

fig.savefig('Figure\\Argand_set_1_Imag.pdf', format="pdf", dpi=1200, bbox_inches='tight')

plt.show()



###############################################################################
###############################################################################
## Critical speed
###############################################################################
###############################################################################



print("u_cr")

print(Speed[indexLambda1])
print(Speed[indexLambda2])

print("Re_cr")

print(Speed[indexLambda1]*beam.data["Sk"]/(2*np.pi))
print(Speed[indexLambda2]*beam.data["Sk"]/(2*np.pi))