# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:43:22 2024

@author: LP275843
"""
import numpy as np

from Main.BeamParameters.Beam import Beam




###############################################################################
###############################################################################
## Beam description
###############################################################################
###############################################################################



# Geometrical Constant

L = 50*10**(-3)
BeamWidth = 100*10**(-3)

###############################################################################
## Blade 1
###############################################################################

BeamThickness = 14.5*10**(-3)

###############################################################################
## Blade 2
###############################################################################

# BeamThickness = 12*10**(-3)

###############################################################################
## Blade 3
###############################################################################

# BeamThickness = 10*10**(-3)

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

###############################################################################
## Blade 1
###############################################################################

zeta = 0.46887022835276243

###############################################################################
## Blade 2
###############################################################################

# zeta = 0.6233043865088962

###############################################################################
## Blade 3
###############################################################################

# zeta = 0.8185053508630611

omega0 = np.sqrt(E*I/((L**4)*rho*S))

xi_T = 0.01

Xi = 0
Zeta = zeta/omega0

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

def CaseStudy():
    return beam