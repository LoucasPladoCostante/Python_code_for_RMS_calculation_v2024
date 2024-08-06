# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:47:29 2024

@author: LP275843
"""

import os
import numpy as np

os.chdir("..\..")

from Main.BeamParameters.BaseFunction import w
from Main.PSD.Matrix import MatrixGeneration
from Main.AddCoef.AddCoefAdim import AddCoefAdim


NFi = 50
model = 'Lagrange'

N = 4
eps = 10**(-8)

def eigValA(beam, guessfqz, NFi, model, N, eps):
    """
    Function giving the two first eigenvalue of the matrix A (cf eq. 52)

    Parameters
    ----------
    beam : Beam
        beam under consideration.
    guessfqz : list of complex
        guess of the firs two eigenmode.
    NFi : int
        parameter for the added mass.
    model : string
        model under consideration.
    N : int
        Half of the size of the matrix A (corresponding to the symbol K in the
        paper but this typology has been removed in the code to prevent
        confusion with the stiffness matrix K).
    eps : float
        precision in the eigenvalue.

    Returns
    -------
    fqz : list of complex
        eigenmode of the matrix A.

    """

    fqz = np.zeros((len(guessfqz)), complex)
    
    
    for g in range(len(guessfqz)):
        conv = False
        while not conv:
            
            Ms, Mm, Mj, Cs, Cc, Cd, Ca, Ks, Kf, Kk, alpha, beta = MatrixGeneration(N, w, **beam.data)
            m0, c0, j0, d0, cu, fd, ku, au, du = AddCoefAdim(guessfqz[g].imag, beam, model, NFi, **beam.data)
            
            M = Ms + (m0 + beam.data["mcyl"]) * Mm + (j0 + beam.data["jcyl"]) * Mj
            C = Cs + (c0 + cu) * Cc + (d0 + du) * Cd + au * Ca
            K = Ks + ((beam.data["mF"]-beam.data["mcyl"]) * beam.data["g"] + fd) * Kf + ku * Kk
            
            MCK = np.zeros((2*N,2*N), complex)
            
            invMK = -np.dot(np.linalg.inv(M),K)
            invMC = -np.dot(np.linalg.inv(M),C)
            
            for i in range(N):
                MCK[i,N+i] = 1
                for j in range(N):
                    MCK[N+i,j] = invMK[i,j]
                    MCK[N+i,N+j] = invMC[i,j]
                    
            eigVal = np.linalg.eig(MCK)[0]
            
            index = 0
            
            for i in range(2*N):
                if np.abs(guessfqz[g]-eigVal[index])>np.abs(guessfqz[g]-eigVal[i]):
                    index = i
                    
            if np.abs(guessfqz[g]-eigVal[index])<eps:
                conv = True
            guessfqz[g] = eigVal[index]
            
        fqz[g] = guessfqz[g]
        
    return (fqz)


