import numpy as np
from scipy.special import gamma, assoc_laguerre, sph_harm
from sympy.physics.wigner import wigner_3j, wigner_6j
from sympy import N
import matplotlib.pyplot as plt
import mcint
import random
import math

def f(x):
    return x[0]*x[1]*x[2]

def sampler():
    while True:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        z = random.uniform(0, 1)
        yield (x, y, z)
        
result, error = mcint.integrate(f, sampler(), measure=1, n=100000000)

print("Result =", result)
print("Error =", error)
