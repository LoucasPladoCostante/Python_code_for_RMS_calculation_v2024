# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:46:47 2024

@author: LP275843
"""
import numpy as np



###############################################################################
###############################################################################
## Description of the class Beam 
###############################################################################
###############################################################################




class Beam:
    
    data = dict()

    def Adim(self):
        self.data["Omega0"] = np.sqrt(self.data["E"]*self.data["I"] /\
            (self.data["rho"]*self.data["S"]*(self.data["L"]**4)))
        self.data["g"] = self.data["G"]/(self.data["L"]\
            *(self.data["Omega0"]**2))
        self.data["mcyl"]=self.data["Mcyl"]/(self.data["L"]*self.data["rho"]\
            *self.data["S"])
        self.data["mF"] = self.data["rhoF"]*np.pi*(self.data["R1"]**2)\
            *self.data["H"]/(self.data["L"]*self.data["rho"]*self.data["S"])
        self.data["h"]  = self.data["H"]/self.data["L"]
        self.data["kappa"] = np.sqrt(self.data["I"]/((self.data["L"]**2)\
            *self.data["S"]))
        self.data["jcyl"] = self.data["Jcyl"]/((self.data["L"]**3)\
            *self.data["rho"]*self.data["S"])
        self.data["xi"] = self.data["Xi"]/self.data["Omega0"]
        self.data["zeta"] = self.data["Zeta"]*self.data["Omega0"]
        self.data["l"]=self.data["H"]/self.data["R1"]
        self.data["epsilon"]=self.data["R2"]/self.data["R1"]
        self.data["Sk"]=((self.data["R2"]-self.data["R1"])**2)\
            *self.data["Omega0"]/self.data["nu"]
        self.data["u"]=4*np.pi*self.data["U"]/(self.data["Omega0"]\
            *(self.data["R2"]-self.data["R1"]))
    
    def __init__(self, L, S, I, R1, H, R2, E, rho, Xi, Zeta, Mcyl, Jcyl, rhoF,\
            nu, U, G):
        
        self.data["L"], self.data["S"], self.data["I"], self.data["R1"],\
            self.data["H"], self.data["R2"], self.data["E"], self.data["rho"],\
            self.data["Xi"], self.data["Zeta"], self.data["Mcyl"],\
            self.data["Jcyl"], self.data["rhoF"], self.data["nu"],\
            self.data["U"], self.data["G"] = \
            L, S, I, R1, H, R2, E, rho, Xi, Zeta, Mcyl, Jcyl, rhoF, nu, U, G
            
        Beam.Adim(self)
        
    def __str__(self):
        return str(self.data)
    
    def strFlowCyl(self):
        return 'l'+str(self.data['l'])+'epsilon'+str(self.data['epsilon'])\
            +'Sk'+str(self.data['Sk'])+'u'+str(self.data['u'])