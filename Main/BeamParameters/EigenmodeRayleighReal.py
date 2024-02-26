# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:11:32 2024

@author: LP275843
"""

import numpy as np
import scipy.linalg as scp

from Main.AddCoef.AddCoefAdim import AddCoefAdim

def prod(omega, beam, model, NFi, mF = 0, mcyl = 0, g = 0, kappa = 0, h = 0, jcyl = 0, **kwargs):
    """
    Evaluation of the matrix A, B, C and the multiplication C exp(A) B as in
    appendix A for the structure in still fluid

    Parameters
    ----------
    omega : float
        Angular frequency.
    beam : Beam
        Beam considered.
    model : string
        model use for the added fluid.
    NFi : int
        number of constant used in order to calculate the added coefficient.
    mF : float, optional
        cf Class Beam. The default is 0.
    mcyl : float, optional
        cf Class Beam. The default is 0.
    g : float, optional
        cf Class Beam. The default is 0.
    kappa : float, optional
        cf Class Beam. The default is 0.
    h : float, optional
        cf Class Beam. The default is 0.
    jcyl : float, optional
        cf Class Beam. The default is 0.
    **kwargs : Beam
        rest of the beam parameters.

    Returns
    -------
    evalProd : np.array
        multiplication C exp(A) B
    A : np.array
        cf Appendix A
    B : np.array
        cf Appendix A
    C : np.array
        cf Appendix A

    """
    m0, c0, j0, d0, cu, fd, ku, au, du = AddCoefAdim(10**80, beam, model, NFi, **beam.data)
    
    A = np.array([[0,        1,                                        0, 0],
                  [0,        0,                                        1, 0],
                  [0,        0,                                        0, 1],
                  [omega**2, 0, ((mF - mcyl) * g) - ((omega * kappa)**2), 0]])
    
    B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
    
    C = np.array([[- (omega ** 2) * (mcyl + m0), ( - (omega ** 2) * (((mcyl + m0) * h / 2) + kappa ** 2)) + (mF - mcyl) * g, 0,    -1],
                  [0,                                                 - (omega**2) * (jcyl + j0 - ((h / 2) * (kappa ** 2))), 1, h / 2]])
    
    evalProd = np.dot(np.dot(C, scp.expm(A)), B)
    
    return evalProd, A, B, C

def prodStr(omega, beam, model, NFi, mF = 0, mcyl = 0, g = 0, kappa = 0, h = 0, jcyl = 0, **kwargs):
    """
    Evaluation of the matrix A, B, C and the multiplication C exp(A) B as in
    appendix A for the structure in air    

    Parameters
    ----------
    omega : float
        Angular frequency.
    beam : Beam
        Beam considered.
    model : string
        model use for the added fluid.
    NFi : int
        number of constant used in order to calculate the added coefficient.
    mF : float, optional
        cf Class Beam. The default is 0.
    mcyl : float, optional
        cf Class Beam. The default is 0.
    g : float, optional
        cf Class Beam. The default is 0.
    kappa : float, optional
        cf Class Beam. The default is 0.
    h : float, optional
        cf Class Beam. The default is 0.
    jcyl : float, optional
        cf Class Beam. The default is 0.
    **kwargs : Beam
        rest of the beam parameters.

    Returns
    -------
    evalProd : np.array
        multiplication C exp(A) B
    A : np.array
        cf Appendix A
    B : np.array
        cf Appendix A
    C : np.array
        cf Appendix A

    """
    A = np.array([[0,        1, 0,                                   0],
                  [0,        0, 1,                                   0],
                  [0,        0, 0,                                   1],
                  [omega**2, 0, (- mcyl * g) - ((omega * kappa)**2), 0]])
        
    B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
    
    C = np.array([[- (omega ** 2) * mcyl, (- (omega ** 2) * ((mcyl * h / 2) + kappa ** 2)) - (mcyl * g), 0,    -1],
                  [0,                     - (omega**2) * (jcyl - ((h / 2) * (kappa ** 2))),              1, h / 2]])
    
    evalProd = np.dot(np.dot(C, scp.expm(A)), B)
    
    return evalProd, A, B, C

def evalDet(omega, beam, model, NFi):
    """
    evaluation of the determinant presented in Appendix A for the structure in
    still fluid

    Parameters
    ----------
    omega : float
        Angular frequency.
    beam : Beam
        Beam considered.
    model : string
        model use for the added fluid.
    NFi : int
        number of constant used in order to calculate the added coefficient.

    Returns
    -------
    determinant : float
        determinant of the product presented inside the function prod.

    """
    
    determinant = np.linalg.det(prod(omega, beam, model, NFi, **beam.data)[0])
    
    return determinant

def evalDetStr(omega, beam, model, NFi):
    """
    evaluation of the determinant presented in Appendix A for the structure in
    air

    Parameters
    ----------
    omega : float
        Angular frequency.
    beam : Beam
        Beam considered.
    model : string
        model use for the added fluid.
    NFi : int
        number of constant used in order to calculate the added coefficient.

    Returns
    -------
    determinant : float
        determinant of the product presented inside the function prodStr.

    """
    
    determinant = np.linalg.det(prodStr(omega, beam, model, NFi, **beam.data)[0])
    
    return determinant

def SearchDicho(Min, Max, beam, model, NFi, structural, eps=10**(-5)):
    """
    function searching when the determinant vanishes using the dichotomy
    algorithm

    Parameters
    ----------
    Min : float
        minimum frequency where a natural frequency can be found.
    Max : float
        maximum frequency where a natural frequency can be found.
    beam : Beam
        Beam considered.
    model : string
        model use for the added fluid.
    NFi : int
        number of constant used in order to calculate the added coefficient.
    structural : bool
        precise if we are looking for the eigenmode of the structure in air or
        still water.
    eps : float, optional
        precision of the result. The default is 10**(-5).

    Returns
    -------
    float
        Natural frequency found with a precision of eps.

    """
    
    omegaMin = Min
    omegaMax = Max
    
    if structural:
        Det = evalDetStr
    else:
        Det = evalDet

    while np.abs(omegaMin-omegaMax)>eps:
        mid = (omegaMin+omegaMax)/2
        if Det(omegaMin, beam, model, NFi) * Det(mid, beam, model, NFi) <= 0:
            omegaMax = mid
        else:
            omegaMin = mid
    return (omegaMin+omegaMax)/2

def seekEigenMode(omegaMin, omegaMax, Resol, beam, model, NFi, structural):
    """
    Function use to seek the natural frequency of the structure

    Parameters
    ----------
    omegaMin : float
        minimum frequency where a natural frequency can be found.
    omegaMax : float
        maximum frequency where a natural frequency can be found.
    Resol : int
        number of evalation we consider in order to seek sign inversion of the
        determinant. The dichotomy algorithm will be used after to seek faster
        and with a higher precision the natural frequency
    beam : Beam
        Beam considered.
    model : string
        model use for the added fluid.
    NFi : int
        number of constant used in order to calculate the added coefficient.
    structural : bool
        precise if we are looking for the eigenmode of the structure in air or
        still water.

    Returns
    -------
    lstGuess : list
        List of the natural frequency.

    """
    
    X = np.linspace(omegaMin, omegaMax, Resol)
    dX = X[1]-X[0]
    lstGuess = []
    
    if structural:
        Det = evalDetStr
    else:
        Det = evalDet
        
    for i in range (Resol):
        if Det(X[i], beam, model, NFi) * Det(X[i]+dX, beam, model, NFi) <= 0:
           lstGuess.append(SearchDicho(X[i], X[i]+dX, beam, model, NFi, structural))
    return lstGuess
    
def getEigenMode(omega, beam, model, NFi, structural):
    """
    fuction returning real value function describing the form of the eigenmode

    Parameters
    ----------
    omega : float
        natural frequency of the mode.
    beam : Beam
        Beam considered.
    model : string
        model use for the added fluid.
    NFi : int
        number of constant used in order to calculate the added coefficient.
    structural : bool
        precise if we are looking for the eigenmode of the structure in air or
        still water.

    Returns
    -------
    w function
        function describing the form of the eigenmode.

    """
    if structural:
        product = prodStr
    else:
        product = prod
    mat, A, B, C = product(omega, beam, model, NFi, **beam.data)
    EigVal, EigVect = np.linalg.eig(mat)
    
    ind = 0
    if np.abs(EigVal[ind]) > np.abs(EigVal[1]):
        ind = 1
    
    def w(x):
        if x<= 1:
            shape = np.dot(np.array([[1, 0, 0, 0]]), np.dot(scp.expm(A*x), np.dot(B, EigVect[:, ind])))
        else:
            shape = np.dot(np.array([[1, 0, 0, 0]]), np.dot(scp.expm(A), np.dot(B, EigVect[:, ind])))\
                + np.dot(np.array([[0, 1, 0, 0]]), np.dot(scp.expm(A), np.dot(B, EigVect[:, ind]))) * (x - 1)
        return shape
    
    return w