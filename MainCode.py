# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:24:25 2024

The following code compute the RMS at a given point of the structure

@author: LP275843
"""
import numpy as np

from Main.BeamParameters.Beam import Beam
from Main.BeamParameters.BaseFunction import w
from Main.PSD.Integrand import Integrand



###############################################################################
###############################################################################
## All parameters have S.I. Units
###############################################################################
###############################################################################



###############################################################################
## Beam description
###############################################################################

# Geometrical Constant

L = 50*10**(-3)                             # Length blade
BeamWidth = 100*10**(-3)                    # Width blade
BeamThickness = 14.5*10**(-3)               # Thickness blade

R1 = 132*10**(-3)                           # Radius inner cylinder
H = 515*10**(-3)                            # Length inner cylinder
CylinderThickness = 4*10**(-3)              # Thickness inner cylinder

R2 = 150*10**(-3)                           # Radius outer cylinder

S = BeamWidth * BeamThickness               # Surface blade
I = ( BeamWidth * BeamThickness ** 3 ) / 12 # Second moment of area

# Material

E = 2.10*10**11                             # Blade Young modulus
rho = 7850                                  # Blade density

# Structural damping

Xi = 0                                      # Mass damping coefficient
Zeta = 5e-05                                # Stifness damping coefficient

# Mass

Mcyl = 30                                   # Mass inner cylinder 
Jcyl = Mcyl*((6*R1**2)+H**2)/12             # Inertia inner cylinder 

# Fluid

rhoF = 1000                                 # Water density
nu = 10**(-6)                               # Water viscosity

# Flow description

U = 2.6                                     # Mean flow velocity

# Gravity

G = 9.81                                    # Gravitational acceleration

# Dimensional axial coordinate to compute the RMS

X = L + H                                   # Position where to compute the RMS

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

def RMSx(X):
    """
    Function calculating the RMS of the beam at a given dimensional coordinate X

    Parameters
    ----------
    x : float
        Dimensionless coordinate where the RMS is computed.

    Returns
    -------
    RMS : float
        Dimensional RMS of the structure given at axial coordinate X.

    """
    x = X/L
    W = np.array([w(i, 0, x) for i in range(NMatrix)])
    RMS = np.sqrt((((beam.data["l"]/(beam.data["h"]*(beam.data["epsilon"]-1\
                )))**2)/(2*np.pi))*np.einsum('i, ij, j->',W, array, W).real)
    return RMS*(beam.data["R2"]-beam.data["R1"])


print("")
print("The dimensional RMS computed is : ")
print(RMSx(X))
