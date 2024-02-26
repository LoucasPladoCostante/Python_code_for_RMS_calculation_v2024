# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:01:34 2024

@author: LP275843
"""

import numpy as np
import sys
from scipy.integrate import dblquad

from Main.PSD.PressureModel import sp

def RMSErcofact2023(beam, omegaMin, omegaMax, mcyl = 0, u = 0, mF = 0, epsilon = 0, l = 0, h = 0, **kwargs):
    """
    RMS computed using the analysis presented in Ercofact written by Maud

    Parameters
    ----------
    beam : Beam
        Beam considered.
    omegaMin : float
        lower bound of the integration.
    omegaMax : float
        upper bound of the integration.
    mcyl : float, optional
        cf Class Beam. The default is 0.
    u : float, optional
        cf Class Beam. The default is 0.
    mF : float, optional
        cf Class Beam. The default is 0.
    epsilon : float, optional
        cf Class Beam. The default is 0.
    l : float, optional
        cf Class Beam. The default is 0.
    h : float, optional
        cf Class Beam. The default is 0.
    **kwargs : Beam
        rest of the beam parameters.

    Returns
    -------
    RMS : float
        RMS calculated cf part 3.4.
    """
    
    eta = 0.05
    resol = 100000
    
    mref = (mcyl*((1/3)+(1/(2*(l**2)))))+(mF*(1/3)*(((epsilon**2)+1)/((epsilon**2)-1)))
    
    lambdac = 0.6
    lambdaeta = 1.6*lambdac
    lambdaphi = 0.29*lambdac
    
    def integrand(omega):
        A__1 = lambdac*u*(epsilon-1)/(4*np.pi*omega*l)
        A__2 = lambdaeta*u*(epsilon-1)/(2*omega*l)
        A__3 = lambdaphi*u*(epsilon-1)/(2*omega)
        
        H = 1 / (np.abs(-(mref * (omega ** 2))+(1j * eta * (np.sqrt(mref) / h) * omega) + (1 / (h ** 2))) ** 2)
        Aleph = sp(omega, **beam.data)*(h**2)/((epsilon-1)**2)
        CaracConst = (1/A__2) + (1j/A__1)
        Ieta = ((2/3)*(CaracConst**(-1)))-(CaracConst**(-2))-(2*(((CaracConst)**(-3))+((CaracConst)**(-4)))*np.exp(-CaracConst))+(2*(CaracConst**(-4)))
        Iphi = 2 * A__3 * (A__3 ** 2 * np.pi - A__3 + np.pi + A__3 * np.exp(-2 /\
                        A__3 * np.pi)) / (A__3 ** 2 + 1) ** 2
        
        return ((H * Aleph * Ieta * Iphi).real)/(2*np.pi)
    
    Omega = np.linspace(omegaMin, omegaMax, resol)
    domega = Omega[1] - Omega[0]
    rms = np.sum(integrand(Omega))*domega
    return np.sqrt(rms)
    
