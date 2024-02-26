# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:50:51 2024

@author: LP275843
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update(mpl.rcParamsDefault)

def FiPerfYeh(omega, l = 0, epsilon = 0, Sk = 0, display = False, **kwargs):
    """
    function computing F1 and F2 as presented in Appendix B using a perfect
    fluid using Yeh model

    Parameters
    ----------
    omega : float
        angular frequency.
    l : float, optional
        cf Class Beam. The default is 0.
    epsilon : float, optional
        cf Class Beam. The default is 0.
    Sk : float, optional
        cf Class Beam. The default is 0.
    display : bool, optional
        display the linear fluid force. The default is False.
    **kwargs : Beam
        rest of the beam parameters.

    Returns
    -------
    F1 : float
        cf Appendix B
    F2 : float
        cf Appendix B
    """
    
    g0 = (epsilon**2 + 1) / (epsilon**2 - 1)
    
    if display:
        
        X = np.linspace(0, 1, 100)
        Y1 = [g0]*len(X)
        Y2 = g0*(X-0.5)
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear mass, $\Re \left\{g_0(1,\eta)\right\}$')
        plt.plot(X,Y1)
        plt.show()
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear mass, $\Re \left\{g_0(\eta-1/2)\right\}$')
        plt.plot(X,Y2)
        plt.show()
        
    
    F1 = g0
    F2 = g0/12
    
    return F1, F2

