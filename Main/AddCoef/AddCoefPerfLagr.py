# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:51:01 2024

@author: LP275843
"""
import scipy as scp
import scipy.special as spe
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["text.usetex"]=True
mpl.rcParams["font.family"]="Helvetica"

def complex_quadrature(func, a, b, **kwargs):
    """
    integration of a function func: R -> C

    Parameters
    ----------
    func : function
    a : float
        lower bound of the integration.
    b : float
        upper bound of the integration.
    **kwargs : 
        Integration Parameters.

    Returns
    -------
    float
        result integration.
    float
        error.

    """
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    
    real_integral = scp.integrate.quad(real_func, a, b, **kwargs)
    imag_integral = scp.integrate.quad(imag_func, a, b, **kwargs)
    
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def w1 (x):
    """
    mode corresponding to a rigid translation
    """
    return 1

def w2 (x):
    """
    mode corresponding to a rigid rotation
    """
    return x-0.5

def fft(f,n):
    """
    function calculated the n-th component of the decomposition of the mode
    as a sum of sinus
    """
    return scp.integrate.quad(lambda x : 2*f(x)* np.sin(np.pi*n*x), 0, 1)[0]

def Mmat(r, gamma):
    """
    M matrix defined as in Appendix B.
    """
    
    M = np.array([[gamma * spe.h1vp(1, gamma * r), gamma * spe.h2vp(1, gamma * r)]])
    return M
    
def StateFunc(n, l, epsilon, Sk):
    """
    Function calculating $\phi_n(r_1)$ as in Appendix B

    Parameters
    ----------
    n : int
        Argument of the function.
    l : float
        cf Class Beam.
    epsilon : float
        cf Class Beam.
    Sk : float
        cf Class Beam.

    Returns
    -------
    $\phi_n(r_1)$ : float
    """
    
    gamma = 1j * n * np.pi * (epsilon-1) / l
    r1 = 1/(epsilon-1)
    r2 = epsilon/(epsilon-1)
    
    Q = np.array([[1j], [0]])
    
    M1 = Mmat(r1, gamma)
    M2 = Mmat(r2, gamma)
    
    M = np.zeros((2, 2), complex)
    for i in range (2):
        M[0, i] = M1[0, i]
        M[1, i] = M2[0, i]
    
    B = np.array([[spe.hankel1(1, gamma * r1), spe.hankel2(1, gamma * r1)]])
    
    foncEtat = np.dot(B,np.linalg.solve(M,Q))
    return foncEtat[0, 0]
    
    

def h(n, l, epsilon, Sk):
    """
    Function calculating $h_n$ as in Appendix B

    Parameters
    ----------
    n : int
        Argument of the float.
    l : float
        cf Class Beam.
    epsilon : float
        cf Class Beam.
    Sk : float
        cf Class Beam.

    Returns
    -------
    h_n : float

    """
    phi = StateFunc(n, l, epsilon, Sk)
    
    return 1j * phi

def FiPerfLagr(N, omega, l = 0, epsilon = 0, Sk = 0, display = False, **kwargs):
    """
    function computing F1 and F2 as presented in Appendix B using a perfect
    fluid using Lagrange model

    Parameters
    ----------
    N : int
        Number of sinus function calculated in the sum.
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
    
    b1 = np.array([fft(w1,n) for n in range (1,N)])
    b2 = np.array([fft(w2,n) for n in range (1,N)])
    
    hn = np.array([h(n, l, epsilon, Sk*omega) for n in range (1,N)])
    
    def g0(bi, eta):
        g0 = 0
        for n in range (1,N):
            g0 += hn[n-1]*bi[n-1]*np.sin(n*np.pi*eta)
        return (epsilon-1)*g0
    
    F1 = complex_quadrature(lambda eta : g0(b1, eta), 0, 1)[0]
    F2 = complex_quadrature(lambda eta : (eta-0.5)*g0(b2, eta), 0, 1)[0]
    
    if display:
        
        X = np.linspace(0, 1, 100)
        Y1, Y2, Y3, Y4 = [], [], [], []
        for x in X:
            Y1.append(g0(b1, x).real)
            Y2.append(g0(b1, x).imag)
            Y3.append(g0(b2, x).real)
            Y4.append(g0(b2, x).imag)
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear mass, $\Re \left\{g_0(1,\eta)\right\}$')
        plt.plot(X,Y1)
        plt.show()
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear damping, $\Im \left\{g_0(1,\eta)\right\}$')
        plt.plot(X,Y2)
        plt.show()
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear mass, $\Re \left\{g_0(\eta-1/2,\eta)\right\}$')
        plt.plot(X,Y3)
        plt.show()
        plt.xlabel(r'Position, $\eta$')
        plt.ylabel(r'Modal self-added linear damping, $\Im \left\{g_0(\eta-1/2,\eta)\right\}$')
        plt.plot(X,Y4)
        plt.show()
    
    
    return F1, F2

