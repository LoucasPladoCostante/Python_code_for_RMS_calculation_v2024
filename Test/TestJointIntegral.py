# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:06:47 2024

@author: LP275843
"""
import time
import numpy as np
import scipy.integrate as integ
import os

os.chdir('..')

from Main.PSD.JointIntegral import I

"""
Function verifying numerically the result presented Appendix E and the time
gained
"""

lambdac = 0.6
lambdaeta = 1.6*lambdac
lambdaphi = 0.29*lambdac

gap = 20

def complex_quadrature(func, a, b, c, d, **kwargs):
    def real_func(x, y):
        return np.real(func(x, y))
    def imag_func(x, y):
        return np.imag(func(x, y))
    real_integral = integ.dblquad(real_func, a, b, c, d, epsabs = 1e-12, **kwargs)
    imag_integral = integ.dblquad(imag_func, a, b, c, d, epsabs = 1e-12, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def XiEta(eta, omega, u, l, epsilon):
    return np.exp(1j*2*np.pi*omega*2*l*eta/(u*(epsilon-1)*lambdac))\
        *np.exp(-omega*2*l*abs(eta)/(u*(epsilon-1)*lambdaeta))
        
def XiPhi(phi, omega, u, l, epsilon):
    return np.exp(-omega*2*abs(phi)/(u*(epsilon-1)*lambdaphi))

def If(omega, u = 0, l = 0, epsilon = 0, **kwargs):
    IEta = complex_quadrature(lambda x, y: XiEta(x-y, omega, u, l, epsilon), 0, 1, lambda x: max(0, x-gap*(lambdaeta*u*(epsilon-1)/(2*omega*l))), lambda x: min(1, x+gap*(lambdaeta*u*(epsilon-1)/(2*omega*l))))[0]
    IPhi = integ.dblquad(lambda x, y: XiPhi(x-y, omega, u, l, epsilon)*np.cos(x)*np.cos(y), 0, 2*np.pi, lambda x: max(0, x-gap*(lambdaphi*u*(epsilon-1)/(2*omega))), lambda x: min(2*np.pi, x+gap*(lambdaphi*u*(epsilon-1)/(2*omega))))[0]
    return IPhi*IEta

def Ifgamma(omega, u = 0, l = 0, epsilon = 0, **kwargs):
    IEta = complex_quadrature(lambda x, y: XiEta(x-y, omega, u, l, epsilon)*(y-0.5), 0, 1, lambda x: max(0, x-gap*(lambdaeta*u*(epsilon-1)/(2*omega*l))), lambda x: min(1, x+gap*(lambdaeta*u*(epsilon-1)/(2*omega*l))))[0]
    IPhi = integ.dblquad(lambda x, y: XiPhi(x-y, omega, u, l, epsilon)*np.cos(x)*np.cos(y), 0, 2*np.pi, lambda x: max(0, x-gap*(lambdaphi*u*(epsilon-1)/(2*omega))), lambda x: min(2*np.pi, x+gap*(lambdaphi*u*(epsilon-1)/(2*omega))))[0]
    return IEta*IPhi

def Igamma(omega, u = 0, l = 0, epsilon = 0, **kwargs):
    IEta = complex_quadrature(lambda x, y: XiEta(y-x, omega, u, l, epsilon)*(x-0.5)*(y-0.5), 0, 1, lambda x: max(0, x-gap*(lambdaeta*u*(epsilon-1)/(2*omega*l))), lambda x: min(1, x+gap*(lambdaeta*u*(epsilon-1)/(2*omega*l))))[0]
    IPhi = integ.dblquad(lambda x, y: XiPhi(y-x, omega, u, l, epsilon)*np.cos(x)*np.cos(y), 0, 2*np.pi, lambda x: max(0, x-gap*(lambdaphi*u*(epsilon-1)/(2*omega))), lambda x: min(2*np.pi, x+gap*(lambdaphi*u*(epsilon-1)/(2*omega))))[0]
    return IEta*IPhi


omega = 5
u = 7
l = 4.3
epsilon = 1.136

tpsa = time.time()
a, b , c = I(omega, u, l, epsilon) 
tpsb = time.time()
print("time analytic (all integral)")
print(tpsb - tpsa)


print("\n\n analytic")
print (a)
tps1 = time.time()
print("\n numeric")
print(If(omega, u, l, epsilon))
tps2 = time.time()
print("\n time numeric")
print(tps2 - tps1)

print("\n\n analytic")
print (b)
tps1 = time.time()
print("\n numeric")
print(Ifgamma(omega, u, l, epsilon))
tps2 = time.time()
print("\n time numeric")
print(tps2 - tps1)

print("\n\n analytic")
print (c)
tps1 = time.time()
print("\n numeric")
print(Igamma(omega, u, l, epsilon))
tps2 = time.time()
print("\n time numeric")
print(tps2 - tps1)