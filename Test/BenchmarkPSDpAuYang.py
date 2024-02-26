# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:28:30 2024

@author: LP275843
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("..")

mpl.rcParams.update(mpl.rcParamsDefault)

from Main.PSD.PressureModel import sp

"""
Function calculating Au-Yang's PSD of the pression s_P 
"""

Resol = 1000
omega = np.linspace(0, 2.5, Resol)
Y = ((2**7)*(np.pi**5))*sp(omega,np.ones(Resol),2*np.ones(Resol),np.ones(Resol),np.ones(Resol))

###############################################################################
## Plot figure
###############################################################################

plt.ylim(bottom=1e-4, top=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'reduced frequency $f=\omega/u$')
plt.ylabel(r'dimonsionless pressure PSD $\aleph(f)$')
plt.xlim(left = 0.005, right = 2.5)
# plt.ylim(bottom = 10**(-4),top=0.310)
plt.plot(omega, Y)
plt.show()