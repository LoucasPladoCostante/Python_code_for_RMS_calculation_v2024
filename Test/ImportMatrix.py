# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:51:56 2024

@author: LP275843
"""

import os

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Matrix import MatrixGeneration

"""
Test display matrix
"""

beam = CaseStudy()
        
print(beam)
N = 5

Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta = MatrixGeneration(N, w, **beam.data)

print("\n\nMs :\n")
print(Ms)
print("\nMm :\n")
print(Mm)
print("\nMj :\n")
print(Mj)
print("\nCs :\n")
print(Cs)
print("\nCc :\n")
print(Cc)
print("\nCd :\n")
print(Cd)
print("\nCa :\n")
print(Ca)
print("\nKs :\n")
print(Ks)
print("\nKf :\n")
print(Kf)
print("\nKk :\n")
print(Kk)
print("\nalpha :\n")
print(alpha)
print("\nbeta :\n")
print(beta)