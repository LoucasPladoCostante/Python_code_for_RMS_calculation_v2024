# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:06:56 2024

@author: LP275843
"""
import scipy as scp
import numpy as np
        
def wij(delta, i, j, derivativei, derivativej, w):
    """
    function calculating the integral of the product of each term related to
    the trial function exposed in Appendix D

    Parameters
    ----------
    delta : int 
        precise if there is a dirac multiplicated to the product 1 if it is the
        case, 0 if not (or anything else).
    i : int
        index trial function.
    j : int
        index trial function.
    derivativei : int
        order of the derivative.
    derivativej : int
        order of the derivative.
    w : function
        test function considerated.

    Returns
    -------
    float
        result of the integrand.

    """
    if delta==1:
        return w(i, derivativei, 1) * w(j, derivativej, 1)
    else:
        return scp.integrate.quad(lambda x: w(i, derivativei, x)*w(j, derivativej, x), 0, 1)[0]

def MatrixGeneration(N, w, h=0, kappa=0, xi=0, zeta=0, **kwargs):
    """
    Computation of the matrix presented in Appendix D

    Parameters
    ----------
    N : int
        Number of test function.
    w : function
        choice of test function.
    h : float, optional
        cf Class Beam. The default is 0.
    kappa : float, optional
        cf Class Beam. The default is 0.
    xi : float, optional
        cf Class Beam. The default is 0.
    zeta : float, optional
        cf Class Beam. The default is 0.
    **kwargs : Beam
        rest of the beam parameters.

    Returns
    -------
    Ms : np.array
        cf Appendix D.
    Mm : np.array
        cf Appendix D.
    Mj : np.array
        cf Appendix D.
    Cs : np.array
        cf Appendix D.
    Cc : np.array
        cf Appendix D.
    Cd : np.array
        cf Appendix D.
    Ca : np.array
        cf Appendix D.
    Ks : np.array
        cf Appendix D.
    Kf : np.array
        cf Appendix D.
    Kk : np.array
        cf Appendix D.
    alpha : np.array
        cf Appendix D.
    beta : np.array
        cf Appendix D.

    """
    Ms = np.zeros((N,N), complex)
    Mm = np.zeros((N,N), complex)
    Mj = np.zeros((N,N), complex)
    
    Cs = np.zeros((N,N), complex)
    Cc = np.zeros((N,N), complex)
    Cd = np.zeros((N,N), complex)
    Ca = np.zeros((N,N), complex)
    
    Ks = np.zeros((N,N), complex)
    Kf = np.zeros((N,N), complex)
    Kk = np.zeros((N,N), complex)
    
    alpha = np.zeros((N,1), complex)
    beta  = np.zeros((N,1), complex)
    
    for i in range (N):
        for j in range (N):
            
            Ms[i,j] = wij(0,i,j,0,0,w) + (kappa**2)*wij(1,i,j,0,1,w)\
                - (kappa**2)*wij(0,i,j,0,2,w)
            Mm[i,j] = (h/2)*wij(1,i,j,1,0,w)+(h/2)*wij(1,i,j,0,1,w)\
                +((h/2)**2)*wij(1,i,j,1,1,w)+wij(1,i,j,0,0,w)
            Mj[i,j] = wij(1,i,j,1,1,w)
            
            Cs[i,j] = 2*xi*wij(0,i,j,0,0,w) + 2*zeta*wij(0,i,j,0,4,w)
            Cc[i,j] = (h/2)*wij(1,i,j,1,0,w) + (h/2)*wij(1,i,j,0,1,w)\
                + ((h/2)**2)*wij(1,i,j,1,1,w) + wij(1,i,j,0,0,w)
            Cd[i,j] = wij(1,i,j,1,1,w)
            Ca[i,j] = wij(1,i,j,0,1,w) + (h/2)*wij(1,i,j,1,1,w)
            
            Ks[i,j] = wij(1,i,j,1,2,w) - wij(1,i,j,0,3,w) + wij(0,i,j,0,4,w)
            Kf[i,j] = -wij(0,i,j,0,2,w) + wij(1,i,j,0,1,w)\
                + (h/2)*wij(1,i,j,1,1,w)
            Kk[i,j] = wij(1,i,j,0,1,w) + (h/2)*wij(1,i,j,1,1,w)
        
        alpha[i,0] = w(i, 0, 1) + (h/2)*w(i, 1, 1)
        beta[i,0] = w(i, 1, 1)
            
            
    return Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta