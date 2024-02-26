# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:06:17 2024

@author: LP275843
"""
import os
import numpy as np

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.EigenmodeRayleighReal import seekEigenMode
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Matrix import MatrixGeneration

"""
Function calculating an adequate value of zeta cf Appendix D
"""

NFi = 50
N = 5

omegaMin = 10**(-5)
omegaMax = 60

Resol = 150

model = "Lagrange"
# model = "Yeh"

beam = CaseStudy()

print(beam)

lst = seekEigenMode(omegaMin, omegaMax, Resol, beam, model, NFi, True)

print(lst)

Xi_T = 0.01
omega_S = lst[1]

beam.data["xi"] = 0
beam.data["zeta"] = 1

Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta = MatrixGeneration(N, w, **beam.data)

M = Ms + (beam.data["mcyl"] * Mm) + (beam.data["jcyl"] * Mj)
K = Ks - beam.data["mcyl"] * beam.data["g"] * Kf

MatMK = np.dot(np.linalg.inv(M), K)

eigVal, eigMod = np.linalg.eig(MatMK)

prod = np.dot(np.linalg.inv(eigMod), np.dot(np.dot(np.linalg.inv(M), Cs/2), eigMod))

index = np.argmin(np.abs((eigVal-(omega_S**2)).real))

zeta = Xi_T * omega_S / prod[index, index]

print("\n\n zeta  :  "+str(zeta)) # dismiss the imaginary part