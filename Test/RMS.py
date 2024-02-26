# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:20:53 2024

@author: LP275843
"""

import os
import numpy as np

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Integrand import Integrand

"""
Function computing the RMS of the displacement at the tip cylinder
"""

beam = CaseStudy()

beam.data["u"] = 0.32

Resol = 1000
NMatrix = 5
NFi = 50
omegaMin = 10**(-5)
omegaMax = 2.5*beam.data["u"]
model = "Lagrange"
# model = "Yeh"

array = Integrand(w, NMatrix, NFi, model, beam, omegaMin, omegaMax, Resol)[0]

RMS = np.sqrt((((beam.data["l"]/(beam.data["h"]*(beam.data["epsilon"]-1)))**2)/(2*np.pi))*np.einsum('ij->', array).real)

print("")
print("RMS = "+str(RMS))