# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:21:38 2024

@author: LP275843
"""
 
import numpy as np

from Main.AddCoef.AddCoefViscLagr import FiViscLagr
from Main.AddCoef.AddCoefViscYeh import FiViscYeh
from Main.AddCoef.AddCoefPerfLagr import FiPerfLagr
from Main.AddCoef.AddCoefPerfYeh import FiPerfYeh
from Main.AddCoef.FrictionFactor import FrictionFactor


def AddCoefAdim(omega, beam, model, NFi, mF=0, epsilon=0, u=0, h=0, l=0, Sk=0, **kwargs):
    """
    function computed the coefficient related to 

    Parameters
    ----------
    omega : float
        dimonsionless angular frequency when the added coefficient is
        calculated.
    beam : Beam
        Beam considered.
    model : string
        model use for the added fluid.
    NFi : int
        Number of sinus function calculated in the sum (for Lagrange viscous
        model).
    mF : float, optional
        cf Class Beam. The default is 0.
    epsilon : float, optional
        cf Class Beam. The default is 0.
    u : float, optional
        cf Class Beam. The default is 0.
    h : float, optional
        cf Class Beam. The default is 0.
    l : float, optional
        cf Class Beam. The default is 0.
    Sk : float, optional
        cf Class Beam. The default is 0.
    **kwargs : Beam
        rest of the beam parameters.

    Raises
    ------
    NameError
        Invalid name for model.

    Returns
    -------
    m0 : float
        cf part 2.3.
    c0 : float
        cf part 2.3.
    j0 : float
        cf part 2.3.
    d0 : float
        cf part 2.3.
    cu : float
        cf part 2.3.
    fd : float
        cf part 2.3.
    ku : float
        cf part 2.3.
    au : float
        cf part 2.3.
    du : float
        cf part 2.3.

    """
    if model == "Lagrange":
        F1perf, F2perf = FiPerfLagr(NFi, omega, **beam.data)
        F1, F2 = F1perf, F2perf
        if Sk*omega<5000:
            F1, F2 = FiViscLagr(NFi, omega, **beam.data)
            
    elif model == "Yeh":
        F1perf, F2perf = FiPerfYeh(omega, **beam.data)
        F1, F2 = F1perf, F2perf
        if Sk*omega<5000:
            F1, F2 = FiViscYeh(omega, **beam.data)
    
    else:
        raise NameError("Unknown model")
        
    CT, CN = FrictionFactor(**beam.data)
    
    m0 = mF*F1.real
    c0 = -mF*omega*F1.imag
    j0 = (h**2)*mF*F2.real
    d0 = -(h**2)*mF*omega*F2.imag
    
    cu = mF*(epsilon-1)*np.abs(u)*CN/((2*np.pi)**2)
    fd = mF*h*((epsilon-1)**2)*(np.abs(u)*u)*CT/(l*np.pi*((4*np.pi)**2))
    ku = mF*h*((epsilon-1)**2)*(np.abs(u)*u)*(CN-CT)/(l*np.pi*((4*np.pi)**2))
    au = 2*(mF*F1perf.real)*h*(epsilon-1)*u/(l*4*np.pi)
    du = mF*(h**2)*(epsilon-1)*np.abs(u)*CN/(48*(np.pi**2))
    
    return m0, c0, j0, d0, cu, fd, ku, au, du