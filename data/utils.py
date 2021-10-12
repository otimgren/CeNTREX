import numpy as np


import numpy as np

#Constants for Boltzmann distribution
T = 6.2 #Kelvin
k_B = 1.38e-23 #J/K
B = 6.66e9*6.63e-34 #Joules
Jmax = 20

def Boltzmann_pop(J, T):
    """
    Function that calculates the population in a single hyperfine level with given J based on the Boltzmann distribution at
    temperature T
    """
    
    #Calculate partition function
    Z = np.sum([4*(2*j+1)*np.exp(-B*j*(j+1)/(k_B*T)) for j in range(0,Jmax+1)])
    
    #Calculate population in hyperfine state with given J
    pop = np.exp(-B*J*(J+1)/(k_B*T))/Z
    
    return pop
