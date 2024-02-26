# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:55:55 2024

@author: LP275843
"""
import numpy as np

def sp(omega, u = 0, epsilon = 0, mF = 0, h = 0, **kwargs):
    """
    Function calculating the PSD at a given point or array 

    Parameters
    ----------
    omega : float or np.array
        cf Class Beam.
    u : float, optional
        cf Class Beam. The default is 0.
    epsilon : float, optional
        cf Class Beam. The default is 0.
    mF : float, optional
        cf Class Beam. The default is 0.
    h : float, optional
        cf Class Beam. The default is 0.
    **kwargs : Beam
        rest of the beam parameters.

    Returns
    -------
    float or np.array
        evaluation of the PSD at a given angular frequency (or array of angular
        frequency)

    """
    fr = omega / u
    return (u**3)*((epsilon-1)**4)*((mF/h)**2)*((fr<0.5)*0.310*np.exp(-6.00*fr)+(fr>=0.5)*0.054*np.exp(-2.52*fr))/((2**7)*(np.pi**5))