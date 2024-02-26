# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:24:25 2024

@author: LP275843
"""
import numpy as np

from Main.BeamParameters.Beam import Beam
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Integrand import Integrand


###############################################################################
## Beam description
###############################################################################

# Geometrical Constant

L = 50*10**(-3)
BeamWidth = 100*10**(-3)
BeamThickness = 14.5*10**(-3)

R1 = 132*10**(-3)
H = 515*10**(-3)
CylinderThickness = 4*10**(-3)

R2 = 150*10**(-3)

S = BeamWidth * BeamThickness
I = ( BeamWidth * BeamThickness ** 3 ) / 12

# Material

E = 2.10*10**11
rho = 7850

# Structural damping

Xi = 0
Zeta = 5e-05

# Mass

Mcyl = 30
Jcyl = Mcyl*((6*R1**2)+H**2)/12

# Fluid

rhoF = 1000
nu = 10**(-6)

# Flow description

U = 2.6

# Other

G = 9.81

beam = Beam(L, S, I, R1, H, R2, E, rho, Xi, Zeta, Mcyl, Jcyl, rhoF, nu, U, G)

###############################################################################
## Numeric constant
###############################################################################

Resol = 1000
NMatrix = 5
NFi = 50
omegaMin = 10**(-5)
omegaMax = 2.5*beam.data["u"]

###############################################################################
## Calculation of the RMS
###############################################################################

model = "Lagrange"
# model = "Yeh"


# Calculation of the integral for the RMS 
array = Integrand(w, NMatrix, NFi, model, beam, omegaMin, omegaMax, Resol)[0]

def RMSx(x):
    """
    Function calculating the RMS of the beam at a given point x

    Parameters
    ----------
    x : float between
        DESCRIPTION.

    Returns
    -------
    RMS : TYPE
        DESCRIPTION.

    """
    W = np.array([w(i, 0, x) for i in range(NMatrix)])
    RMS = np.sqrt((((beam.data["l"]/(beam.data["h"]*(beam.data["epsilon"]-1\
                )))**2)/(2*np.pi))*np.einsum('i, ij, j->',W, array, W).real)
    return RMS

x = 1
print("")
print(RMSx(x))