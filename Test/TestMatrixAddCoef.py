# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:08:28 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.PSD.Matrix import MatrixGeneration
from Main.BeamParameters.BaseFunction import w
from Main.AddCoef.AddCoefAdim import AddCoefAdim

"""
Function calculating the dimensional natural frequency of the structure in 
still water using the matrix defined in Appendix D in order to compare with the 
solution obtained using Appendix A
"""

beam = CaseStudy()

N = 5 # NB for this specific purpose, it is recommended to not working with a lot
# of function since np.linalg.eig is specialised in looking for high frequency
# rather than the lower - it becomes difficult to have the first natural frequency -
Resol = 1000

omega = 10**80 # Access to the perfect model
model = "Lagrange"
# model = "Yeh"
NFi = 50

m0, c0, j0, d0, cu, fd, ku, au, du = AddCoefAdim(omega, beam, model, NFi, **beam.data)

Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta = MatrixGeneration(N, w, **beam.data)

M = Ms + (beam.data["mcyl"] + m0) * Mm + (beam.data["jcyl"] + j0) * Mj

K = Ks + ((beam.data["mF"] - beam.data["mcyl"]) * beam.data["g"]) * Kf
# K = Ks + ((beam.data["mF"] - beam.data["mcyl"]) * beam.data["g"] + fd) * Kf + ku * Kk

Adimfreq = np.sqrt(np.linalg.eig(np.dot(np.linalg.inv(M),K))[0])
freq = Adimfreq*beam.data["Omega0"]/(2*np.pi)
eigenVect = np.linalg.eig(np.dot(np.linalg.inv(M),K))[1]

lstEigenVect = []

X = np.linspace(0,1.5, Resol)
for i in range (N):
    Y = np.zeros(Resol,complex)
    for x in range (Resol):
        Y[x] = np.sum(np.array([eigenVect[j,i]*w(j,0,X[x]).real for j in range (N)]))
    lstEigenVect.append(Y)
    plt.plot(X,Y.real)
    plt.show()

print("dimensionless natural frequency")
print(Adimfreq)
print("dimensionalised natural frequency (in Hz)")
print(freq)

#### A selection of the first eigenvalues lead to

Eig = [3, 4, 2, 1, 0]
plt.xlim(left = 0, right=1.5)
for i in Eig:
    plt.plot(X,(lstEigenVect[i].real/np.max(np.abs(lstEigenVect[i]))) * 
             np.sign(lstEigenVect[i][Resol//2].real))

plt.show()
