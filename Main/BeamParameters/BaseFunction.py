# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:20:53 2024

@author: LP275843
"""

from Main.BeamParameters.Geometry import CaseStudy

beam = CaseStudy()

def w(i, derivative, x):
    """
    trial function used for the construction of the matrix

    Parameters
    ----------
    i : int
        index of the test function.
    derivative : int
        order of the derivative.
    x : int
        point of the evaluation.

    Raises
    ------
    NameError
        derivative of too high order is required.

    Returns
    -------
    $w_i^{(derivative)}(x)$ : float
        evaluate $w_i^{(derivative)}(x)$.

    """
    if x<=1:
        if derivative == 0:
            res = x**(i+2)
        elif derivative == 1:
            res = (i+2)*x**(i+1)
        elif derivative == 2:
            res = (i+2)*(i+1)*x**i
        elif derivative == 3:
            if i==0:
                res = 0
            else:
                res = (i+2)*(i+1)*i*x**(i-1)
        elif derivative == 4:
            if i<=1:
                res = 0
            else:
                res = (i+2)*(i+1)*i*(i-1)*x**(i-2)
        else:
            raise NameError("Error trial fonction")
    else:
        if derivative == 0:
            res = 1 + (i+2)*(x-1)
        elif derivative == 1:
            res = i+2
        elif derivative > 1:
            res = 0
        else:
            raise NameError("Error trial fonction")
    return res/(1+(i+2)*beam.data["h"])