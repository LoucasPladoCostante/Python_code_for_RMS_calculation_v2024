# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:02:28 2024

@author: LP275843
"""
import os
import sys
import numpy as np

from Main.PSD.Matrix import MatrixGeneration
from Main.PSD.JointIntegral import I
from Main.PSD.PressureModel import sp
from Main.AddCoef.AddCoefAdim import AddCoefAdim

def Integrand(w, NMatrix, NFi, model, beam, omegaMin, omegaMax, Resol):
    """
    function calculating the integral of the transfer function [H][H]*s_P

    Parameters
    ----------
    w : function
        trial function used.
    NMatrix : int
        Number of trial function used.
    NFi : int
        Number of sinus function calculated in the sum (for Lagrange viscous
        model).
    model : string
        model use for the added fluid.
    beam : Beam
        Beam considered.
    omegaMin : float
        lower bound of the integration.
    omegaMax : float
        upper bound of the integration.
    Resol : int
        nuber of step inside the integration.

    Returns
    -------
    integrand : np.array
        Integral of the transfer function in order to calculate the RMS.
    matH : np.array
        array containing all the matrix [H][H]*s_P separated with a linear 
        regular space

    """
    Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta = MatrixGeneration(NMatrix, w, **beam.data)
    
    def matB(omega):
        m0, c0, j0, d0, cu, fd, ku, au, du = AddCoefAdim(omega, beam, model, NFi, **beam.data)
        M = Ms + (m0 + beam.data["mcyl"]) * Mm + (j0 + beam.data["jcyl"]) * Mj
        C = Cs + (c0 + cu) * Cc + (d0 + du) * Cd + au * Ca
        K = Ks + ((beam.data["mF"]-beam.data["mcyl"]) * beam.data["g"] + fd) * Kf + ku * Kk
        return - (omega**2) * M + 1j * omega * C + K
    
    def matTransfert(omega):
        evalIf, evalIfgamma, evalIgamma = I(omega, **beam.data)
        
        B = matB(omega)
        H1 = np.conj(np.dot(np.linalg.inv(B), alpha))
        H2 = np.conj(np.dot(np.linalg.inv(B), beta))
        
        H = (beam.data["h"]**4)/(beam.data["l"]**2) * np.dot(H1, np.transpose(np.conj(H1))) * evalIf +\
            (beam.data["h"]**5)/(beam.data["l"]**2) * np.dot(H1, np.transpose(np.conj(H2))) * evalIfgamma +\
            (beam.data["h"]**5)/(beam.data["l"]**2) * np.dot(H2, np.transpose(np.conj(H1))) * np.conj(evalIfgamma) +\
            (beam.data["h"]**6)/(beam.data["l"]**2) * np.dot(H2, np.transpose(np.conj(H2))) * evalIgamma
        return H
    
    Omega = np.linspace(omegaMin, omegaMax, Resol)
    domega = Omega[1]-Omega[0]
    lstH = []
    for i in range (Resol):  
        sys.stdout.write('\r'+str(100*i//Resol)+" %")
        lstH.append(matTransfert(Omega[i]))
    matH=np.array(lstH)
    Sp = sp(Omega, **beam.data)
    
    integrand = np.zeros((NMatrix,NMatrix), complex)
    integrand = np.einsum("ijk,i->jk", matH, Sp) * domega
    
    return integrand, matH

def Integrandlog(w, NMatrix, NFi, model, beam, omegaMin, omegaMax, Resol):
    """
    function returning the product [H][H]*s_P for each omega, equaly space on a 
    log scale

    Parameters
    ----------
    w : function
        trial function used.
    NMatrix : int
        Number of trial function used.
    NFi : int
        Number of sinus function calculated in the sum (for Lagrange viscous
        model).
    model : string
        model use for the added fluid.
    beam : Beam
        Beam considered.
    omegaMin : float
        lower bound of the integration.
    omegaMax : float
        upper bound of the integration.
    Resol : int
        nuber of step inside the range of calculation.

    Returns
    -------
    matH : np.array
        array containing all the matrix [H][H]*s_P separated with an regular 
        space on a log scale

    """
    Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta = MatrixGeneration(NMatrix, w, **beam.data)
    
    def matB(omega):
        m0, c0, j0, d0, cu, fd, ku, au, du = AddCoefAdim(omega, beam, model, NFi, **beam.data)
        M = Ms + (m0 + beam.data["mcyl"]) * Mm + (j0 + beam.data["jcyl"]) * Mj
        C = Cs + (c0 + cu) * Cc + (d0 + du) * Cd + au * Ca
        K = Ks + ((beam.data["mF"]-beam.data["mcyl"]) * beam.data["g"] + fd) * Kf + ku * Kk
        return - (omega**2) * M + 1j * omega * C + K
    
    def matTransfert(omega):
        evalIf, evalIfgamma, evalIgamma = I(omega, **beam.data)
        
        B = matB(omega)
        H1 = np.conj(np.dot(np.linalg.inv(B), alpha))
        H2 = np.conj(np.dot(np.linalg.inv(B), beta))
        
        H = (beam.data["h"]**4)/(beam.data["l"]**2) * np.dot(H1, np.transpose(np.conj(H1))) * evalIf +\
            (beam.data["h"]**5)/(beam.data["l"]**2) * np.dot(H1, np.transpose(np.conj(H2))) * evalIfgamma +\
            (beam.data["h"]**5)/(beam.data["l"]**2) * np.dot(H2, np.transpose(np.conj(H1))) * np.conj(evalIfgamma) +\
            (beam.data["h"]**6)/(beam.data["l"]**2) * np.dot(H2, np.transpose(np.conj(H2))) * evalIgamma
        return H
    
    Omega = np.logspace(np.log10(omegaMin), np.log10(omegaMax), Resol)
    lstH = []
    for i in range (Resol):  
        sys.stdout.write('\r'+str(100*i//Resol)+" %")
        lstH.append(matTransfert(Omega[i]))
    matH=np.array(lstH)
    
    return matH