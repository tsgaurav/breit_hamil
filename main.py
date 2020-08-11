import numpy as np
from scipy.special import gamma, assoc_laguerre, sph_harm
from sympy.physics.wigner import wigner_3j, wigner_6j
from sympy import N
import matplotlib.pyplot as plt
import mcint
import random
import math
import time
import py3nj
import sys
from scipy.stats import expon
from scipy.stats import gamma as gam
from operators import *
from me import *


def anti_symmetrize(r_sph_doub):

    n1, l1, j1, n2, l2, j2 = ab
    n3, l3, j3, n4, l4, j4 = cd
    J, M = JM
    norm = 1.
    state_sum = 0.
    r1, theta1, phi1, r2, theta2, phi2 = r_sph_doub

    if (n1, l1, j1) == (n2, l2, j2): norm = norm*np.sqrt(0.5)
    if (n3, l3, j3) == (n4, l4, j4): norm = norm*np.sqrt(0.5)
    
    state_sum+=me_term(r_sph_doub, n1, l1, j1, n2, l2, j2, n3, l3, j3, n4, l4, j4, J, M, zeta)
    
    n2, l2, j2, n1, l1, j1 = ab
    n3, l3, j3, n4, l4, j4 = cd
    
    state_sum+=me_term(r_sph_doub, n1, l1, j1, n2, l2, j2, n3, l3, j3, n4, l4, j4, J, M, zeta)*(-1)**(j1+j2-J-1)

    n1, l1, j1, n2, l2, j2 = ab
    n4, l4, j4, n3, l3, j3 = cd

    state_sum+=me_term(r_sph_doub, n1, l1, j1, n2, l2, j2, n3, l3, j3, n4, l4, j4, J, M, zeta)*(-1)**(j3+j4-J-1)

    n2, l2, j2, n1, l1, j1 = ab
    n4, l4, j4, n3, l3, j3 = cd

    state_sum+=me_term(r_sph_doub, n1, l1, j1, n2, l2, j2, n3, l3, j3, n4, l4, j4, J, M, zeta)*(-1)**(j1+j2+j3+j4)

    return state_sum*norm*0.5/gam.pdf([r1, r2], a_gam, scale=scale).prod()





def sampler_three():
    while True:
        #u = np.random.gamma(a_gam, scale=scale)
        u = random.uniform(0, 10)
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, math.pi)
        yield (u, theta, phi)

def sampler_six():
    while True:
        #u1 = random.uniform(0, 10)
        u1 = np.random.gamma(a_gam, scale=scale)
        theta1 = random.uniform(0, 2*math.pi)
        phi1 = random.uniform(0, math.pi)
        #u2 = random.uniform(0, 10)
        u2 = np.random.gamma(a_gam, scale=scale)
        theta2 = random.uniform(0, 2*math.pi)
        phi2 = random.uniform(0, math.pi)
        yield (u1, theta1, phi1, u2, theta2, phi2)

domainsize3 = 10*2*math.pi**2
domainsize6 = 4*math.pi**4

nmc = 20000

#scale = 2#float(sys.argv[1])

#a_gam = 2#float(sys.argv[1])

me_term = specific_ms

ab = [2, 0, 1/2, 2, 0, 1/2]
cd = [0, 1, 1/2, 0, 1, 1/2]
JM = [0, 0]
zeta = 1

start = time.time()
#result, error = mcint.integrate(normal_ms_alt, sampler_three(), measure=domainsize3, n=nmc)
#result, error = mcint.integrate(anti_symmetrize, sampler_six(), measure=domainsize6, n=nmc)

#print("Scale =", scale)
#print("Result = ", result.real)
#print("Error estimate =", error)
#print("Time =", time.time()-start)

scales = np.array([1, 1.5, 2, 2.5, 3])
gams = np.array([1, 1.5, 2, 2.5, 3])
results = np.zeros(shape=(len(scales), len(gams)))
errors = np.zeros(shape=(len(scales), len(gams)))


for i, scale in enumerate(scales):
    for j, a_gam in enumerate(gams):
        result, error = mcint.integrate(anti_symmetrize, sampler_six(), measure=domainsize6, n=nmc)
        results[i, j] = result
        errors[i, j] = error

np.save("Results5533.npy", results)
np.save("Errors5533.npy", errors)
