# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:06:47 2024

@author: LP275843
"""
import numpy as np

lambdac = 0.6
lambdaeta = 1.6*lambdac
lambdaphi = 0.29*lambdac

def I(omega, u = 0, l = 0, epsilon = 0, **kwargs):
    """
    Compute the joint acceptance as in Appendix E

    Parameters
    ----------
    omega : float
        angular frequecy considered.
    u : float, optional
        cf Class Beam. The default is 0.
    l : float, optional
        cf Class Beam. The default is 0.
    epsilon : float, optional
        cf Class Beam. The default is 0.
    **kwargs : Beam
        rest of the beam parameters.

    Returns
    -------
    float
        If in Appendix E.
    float
        Ifgamma in Appendix E.
    float
        Igamma in Appendix E.

    """
    A__1 = lambdac*u*(epsilon-1)/(4*np.pi*omega*l)
    A__2 = lambdaeta*u*(epsilon-1)/(2*omega*l)
    A__3 = lambdaphi*u*(epsilon-1)/(2*omega)
    resultf = 2 * ((complex(0, -1) * A__1 * A__2 ** 2 + A__1 ** 2 * A__2 / 2 -\
                    A__2 ** 3 / 2) * np.exp((complex(0, -1) * A__2 - A__1) / A__1\
                    / A__2) + A__2 * (complex(0, 1) * A__1 * A__2 + A__1 ** 2 / 2\
                    - A__2 ** 2 / 2) * np.exp((complex(0, 1) * A__2 - A__1) / A__1\
                    / A__2) - A__1 ** 2 * A__2 + A__2 ** 3 + A__1 ** 2 + A__2 ** \
                    2) * A__2 * A__1 ** 2 / (A__1 ** 2 + A__2 ** 2) ** 2
    resultfg = -3 * (((-A__2 / 3 - 0.1e1 / 0.6e1) * A__1 ** 4 + complex(0, 1) * \
                    A__2 * (A__2 + 0.1e1 / 0.3e1) * A__1 ** 3 + A__1 ** 2 * A__2\
                    ** 3 + complex(0, -0.1e1 / 0.3e1) * (A__2 - 1) * A__2 ** 3 *\
                    A__1 + A__2 ** 4 / 6) * np.exp((complex(0, -1) * A__2 - A__1)\
                    / A__1 / A__2) + ((A__2 / 3 + 0.1e1 / 0.6e1) * A__1 ** 4 +\
                    complex(0, 1) * A__2 * (A__2 + 0.1e1 / 0.3e1) * A__1 ** 3 -\
                    A__1 ** 2 * A__2 ** 3 + complex(0, -0.1e1 / 0.3e1) * (A__2 -\
                    1) * A__2 ** 3 * A__1 - A__2 ** 4 / 6) * np.exp((complex(0, 1)\
                    * A__2 - A__1) / A__1 / A__2) + complex(0, -2) * ((A__2 -\
                    0.1e1 / 0.3e1) * A__1 ** 2 - A__2 ** 2 * (A__2 + 1) / 3) *\
                    A__2 * A__1) * A__2 ** 2 * A__1 ** 2 / (A__1 ** 2 + A__2 **\
                    2) ** 3
    resultg = 4 * (((complex(0, -0.1e1 / 0.4e1) * A__1 - A__1 ** 2 / 4 + 0.1e1 /\
                    0.16e2) * A__2 ** 6 - (complex(0, 1) * A__1 ** 2 + complex(0,\
                    -0.1e1 / 0.8e1) - 0.3e1 / 0.4e1 * A__1) * A__1 * A__2 ** 5 +\
                    (complex(0, 1) * A__1 + 3 * A__1 ** 2 + 0.1e1 / 0.8e1) * A__1\
                    ** 2 * A__2 ** 4 / 2 + A__1 ** 3 * (complex(0, 1) * A__1 ** 2\
                    + complex(0, 0.1e1 / 0.4e1) + A__1 / 2) * A__2 ** 3 +\
                    (complex(0, 0.3e1 / 0.4e1) * A__1 ** 5 - A__1 ** 6 / 4 - A__1\
                    ** 4 / 16) * A__2 ** 2 + (complex(0, 1) - 2 * A__1) * A__1 **\
                    5 * A__2 / 8 - A__1 ** 6 / 16) * A__2 * np.exp((complex(0, -1)\
                    *A__2 - A__1) / A__1 / A__2) - A__2 * ((A__1 ** 2 / 4 +\
                    complex(0, -0.1e1 / 0.4e1) * A__1 - 0.1e1 / 0.16e2) * A__2 **\
                    6 - A__1 * (complex(0, 1) * A__1 ** 2 + complex(0, -0.1e1 /\
                    0.8e1) + 0.3e1 / 0.4e1 * A__1) * A__2 ** 5 + (complex(0, 1) *\
                    A__1 - 3 * A__1 ** 2 - 0.1e1 / 0.8e1) * A__1 ** 2 * A__2 ** 4\
                    / 2 + A__1 ** 3 * (-A__1 / 2 + complex(0, 1) * A__1 ** 2 + \
                    complex(0, 0.1e1 / 0.4e1)) * A__2 ** 3 + (A__1 ** 4 / 16 + \
                    A__1 ** 6 / 4 + complex(0, 0.3e1 / 0.4e1) * A__1 ** 5) * A__2\
                    ** 2 + (2 * A__1 + complex(0, 1)) * A__1 ** 5 * A__2 / 8 + \
                    A__1 ** 6 / 16) * np.exp((complex(0, 1) * A__2 - A__1) / A__1\
                    / A__2) + (0.1e1 / 0.8e1 + A__1 ** 2 / 2) * A__2 ** 7 + A__2\
                    ** 6 / 24 + (-3 * A__1 ** 4 + A__1 ** 2 / 8) * A__2 ** 5 +\
                    A__1 ** 2 * A__2 ** 4 / 8 + (A__1 ** 6 / 2 - A__1 ** 4 / 8) *\
                    A__2 ** 3 + A__1 ** 4 * A__2 ** 2 / 8 - A__2 * A__1 ** 6 / 8 +\
                    A__1 ** 6 / 24) * A__2 * A__1 ** 2 / (A__1 ** 2 + A__2 ** 2) \
                    ** 4
    resultphi = 2 * A__3 * (A__3 ** 2 * np.pi - A__3 + np.pi + A__3 * np.exp(-2 /\
                    A__3 * np.pi)) / (A__3 ** 2 + 1) ** 2
    return resultf * resultphi, resultfg * resultphi , resultg * resultphi


