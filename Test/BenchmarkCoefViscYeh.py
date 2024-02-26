# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:43:38 2024

@author: LP275843
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("..")

mpl.rcParams.update(mpl.rcParamsDefault)

from Main.AddCoef.AddCoefViscYeh import FiViscYeh


"""
Test verifying the model of Yeh for a viscous fluid
"""


Resol = 50

l = -1
omega = 1
Epsilon = np.logspace(0.01, np.log10(20), Resol)
SK = [10, 50, 100, 500, 5000]
# Paidousis FSI Slender Structure and Axial Flow Vol 1 Chap 2 
# SK = [10, 50, 100, 500, 5000, 50000]
# Prec = [100, 100, 100, 200, 500, 1500]

Y1, Y2 = np.zeros((len(SK),Resol)), np.zeros((len(SK),Resol))

for i in range (len(SK)):
    sys.stdout.write('\r'+'S = '+str(SK[i])+"\n")
    for j in range (len(Epsilon)):
        sys.stdout.write('\r'+str(100*j//Resol)+" %")
        f1, f2 = FiViscYeh(omega, l, Epsilon[j], SK[i]*(Epsilon[j]-1)**2, False)
        Y1[i,j] = f1.real 
        Y2[i,j] = -f1.imag
        

###############################################################################
## Plot figure
###############################################################################



plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Radius ratio, $\varepsilon$')
plt.ylabel(r'Modal self-added mass, $\Re \left\{F^{(1)}\right\}$')  
plt.xlim(left = 1, right = 20)
plt.ylim(bottom = 1, top = 20)
for i in range (len(SK)):      
    plt.plot(Epsilon, Y1[i,:], label = r'S = $\Omega R_1^2/\nu =$'+str(SK[i]))
plt.legend()
plt.show()

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Radius ratio, $\varepsilon$')
plt.ylabel(r'Modal self-added damping, $-\Im \left\{F^{(1)}\right\}$')    
plt.xlim(left = 1, right = 20)
plt.ylim(bottom = 0.01, top = 5)
for i in range (len(SK)):      
    plt.plot(Epsilon, Y2[i,:], label = r'S = $\Omega R_1^2/\nu =$'+str(SK[i]))
plt.legend()
plt.show()

FiViscYeh(omega, l, 1.1, 250, True)