# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:52:27 2024

@author: LP275843
"""

import os

os.chdir("..")

from Main.BeamParameters.Geometry import CaseStudy

"""
Test import beam and display caracteristic
"""

beam = CaseStudy()

print(beam)

