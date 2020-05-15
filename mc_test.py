import numpy as np
from scipy.special import gamma, assoc_laguerre, sph_harm
from sympy.physics.wigner import wigner_3j, wigner_6j
from sympy import N
import matplotlib.pyplot as plt
import mcint
import random
import math
from scipy import integrate
import time


def f(r_pol):
    u, theta = r_pol
    return -np.log(u)*np.exp(-(np.log(u)**2))/u

def sampler():
    while True:
        u = random.uniform(0, 1)
        theta = random.uniform(0, 2*math.pi)
        yield (u, theta)

domainsize = 2*math.pi

nmcs = []
elapsed = []
errors = []

for nmc in [5**10]:
    start = time.time()
    result,trash = mcint.integrate(f, sampler(), measure=domainsize, n=nmc)
    nmcs.append(nmc)
    elapsed.append(time.time()-start)
    errors.append(result-np.pi)


print(errors)
