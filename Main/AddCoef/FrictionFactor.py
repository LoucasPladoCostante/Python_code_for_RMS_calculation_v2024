# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:35:14 2024

@author: LP275843
"""
import numpy as np

def FrictionFactor(u = 0, Sk = 0, **kwargs):
    """
    Function computing the friction coefficient $C_N$ and $C_T$

    Parameters
    ----------
    u : float, optional
        cf Class Beam. The default is 0.
    Sk : float, optional
        cf Class Beam. The default is 0.
    **kwargs : Beam
        rest of the beam parameters.

    Returns
    -------
    CT : float
        cf part 3.2
    CN : float
        cf part 3.2

    """
    Cf = (1.14-((2/np.log(10))*np.log(21.25/((np.abs(u)*Sk/(2*np.pi))**0.9))))**(-2)
    CT = np.pi*Cf/4
    CN = CT
    return CT, CN
