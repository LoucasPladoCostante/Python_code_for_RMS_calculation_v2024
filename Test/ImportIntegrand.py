# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:20:53 2024

@author: LP275843
"""

import os

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Integrand import Integrand

"""
Test computation of the integral
"""

beam = CaseStudy()

Resol = 100
NMatrix = 30
NFi = 50
model = "Lagrange"

omegaMin = 10**(-5)
omegaMax = 2.5*beam.data["u"]
# model = "Yeh"

print(Integrand(w, NMatrix, NFi, model, beam, omegaMin, omegaMax, Resol)[0])