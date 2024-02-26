# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:52:17 2024

@author: LP275843
"""

import scipy.special as spe
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update(mpl.rcParamsDefault)

# maximum SK omega = 5000

def FiViscYeh(omega, l = 0, epsilon = 0, Sk = 0, display = False, **kwargs):
    """
    function computing F1 and F2 as presented in Appendix B using a viscous
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
    
    beta1 = (1-1j)*np.sqrt(2*omega*Sk)/(2*(epsilon-1))
    beta2 = epsilon*beta1
        
    M1 = np.array([[1, 1,             spe.hankel1(1, beta1),             spe.hankel2(1, beta1)],
                   [0, 1, (1/epsilon)*spe.hankel1(1, beta2), (1/epsilon)*spe.hankel2(1, beta2)],
                   [2, 2,       beta1*spe.hankel1(0, beta1),       beta1*spe.hankel2(0, beta1)],
                   [0, 2,       beta1*spe.hankel1(0, beta2),       beta1*spe.hankel2(0, beta2)]])
    
    M2 = np.array([[1             , 1,             spe.hankel1(1, beta1),             spe.hankel2(1, beta1)],
                   [(1/epsilon)**2, 1, (1/epsilon)*spe.hankel1(1, beta2), (1/epsilon)*spe.hankel2(1, beta2)],
                   [0             , 2,       beta1*spe.hankel1(0, beta1),       beta1*spe.hankel2(0, beta1)],
                   [0             , 2,       beta1*spe.hankel1(0, beta2),       beta1*spe.hankel2(0, beta2)]])
    
    det1 = np.linalg.det(M1)
    det2 = np.linalg.det(M2)
    
    a = -(det1/det2)
    g0 = -(1+(2*a))

    F1 = g0
    F2 = g0/12
    
    if display:
        
        X = np.linspace(0, 1, 100)
        Y1 = [g0.real]*len(X)
        Y2 = [-g0.imag]*len(X)
        Y3 = g0.real*(X-0.5)
        Y4 = -g0.imag*(X-0.5)
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear mass, $\Re \left\{g_0\right\}$')
        plt.plot(X,Y1)
        plt.show()
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear damping, $-\Im \left\{g_0\right\}$')
        plt.plot(X,Y2)
        plt.show()
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear mass, $\Re \left\{g_0(\eta-1/2)\right\}$')
        plt.plot(X,Y3)
        plt.show()
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear damping, $-\Im \left\{g_0(\eta-1/2)\right\}$')
        plt.plot(X,Y4)
        plt.show()
    
    return F1, F2

