# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:43:39 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.PSD.Matrix import MatrixGeneration
from Main.BeamParameters.BaseFunction import w

"""
Function calculating the dimensional natural frequency of the structure in air
using the matrix defined in Appendix D in order to compare with the solution
obtained using Appendix A
"""

beam = CaseStudy()

N = 5 # NB for this specific purpose, it is recommended to not working with a lot
# of function since np.linalg.eig is specialised in looking for high frequency
# rather than the lower - it becomes difficult to have the first natural frequency -
Resol = 100

Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta = MatrixGeneration(N, w, **beam.data)

M = Ms + beam.data["mcyl"] * Mm + beam.data["jcyl"] * Mj

K = Ks + (beam.data["mF"] - beam.data["mcyl"]) * beam.data["g"] * Kf

freq = np.sqrt(np.linalg.eig(np.dot(np.linalg.inv(M),K))[0])*beam.data["Omega0"]/(2*np.pi)
eigenVect = np.linalg.eig(np.dot(np.linalg.inv(M),K))[1]

lstEigenVect = []

X = np.linspace(0,1.2, Resol)
for i in range (N):
    Y = np.zeros(Resol,complex)
    for x in range (Resol):
        Y[x] = np.sum(np.array([eigenVect[j,i]*w(j,0,X[x]).real for j in range (N)]))
    lstEigenVect.append(Y)
    plt.plot(X,Y)
    plt.show()

print("dimensionalised natural frequency (in Hz)")
print(freq)

#### A selection of the first eigenvalues lead to

Eig = [3, 4, 2, 1, 0]

for i in Eig:
    plt.plot(X,np.sign(lstEigenVect[i][Resol//2])*lstEigenVect[i]/np.max(np.abs(lstEigenVect[i])))

plt.show()