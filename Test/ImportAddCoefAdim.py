# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:23:59 2024

@author: LP275843
"""

import os

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.AddCoef.AddCoefAdim import AddCoefAdim


"""
Function computing the added coefficient at a given frequecy omega
"""

beam = CaseStudy()

omega = 100
model = "Lagrange"
# model = "Yeh"
NFi = 50

m0, c0, j0, d0, cu, fd, ku, au, du = AddCoefAdim(omega, beam, model, NFi, **beam.data)

print("m0 : "+str(m0))
print("c0 : "+str(c0))
print("j0 : "+str(j0))
print("d0 : "+str(d0))
print("cu : "+str(cu))
print("fd : "+str(fd))
print("ku : "+str(ku))
print("au : "+str(au))
print("du : "+str(du))