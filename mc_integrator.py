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


def thj(j1, j2, j3, m1, m2, m3):
    """
    3-j symbol
    ( j1 j2 j3 )
    ( m1 m2 m3 )
    """
    #return wigner3j(j1,j2,j3,m1,m2,m3)
    return py3nj.wigner3j(int(2*j1),int(2*j2),int(2*j3),int(2*m1),int(2*m2),int(2*m3))

def cgc(l, s, j, ml, ms, m):
    return (-1)**(-l+s-m)*np.sqrt(2*j+1)*thj(l, s, j, ml, ms, -m)

def spinor(ms):
    if ms==1/2:
        return np.array([1.,0.])
    elif ms==-1/2:
        return np.array([0.,1.])

def laguerre_wave_function(x, n, l):
    zeta = 1
    """
    Laguerre function, see [A. E. McCoy and M. A. Caprio, J. Math. Phys. 57, (2016).] for details
    """
    eta = -2.0 * np.log(x) / zeta
    return np.sqrt(2.0 * gamma(n+1) / (zeta * gamma(n+2*l+3)) ) * 2.0 * eta**l *\
              np.exp(-0.5*eta) * assoc_laguerre(eta, n, 2*l+2) / zeta

def make_state(r_sph, n, l, j, mj):
    r, theta, phi = r_sph
    state_sum = 0.
    for ml in np.arange(-l, l+1, 1):  #check limits
            state_sum= state_sum+np.array([cgc(l, 1/2, j, ml, 1/2, mj)*sph_harm(ml, l, theta, phi),cgc(l, 1/2, j, ml, -1/2, mj)*sph_harm(ml, l, theta, phi)])
    return laguerre_wave_function(r, n, l)*state_sum


def f(r_sph1):
    u, theta, phi = r_sph1
    #r_sph1 = np.array([u, theta, phi])
    #r_sph2 = cart_to_sph([x2, y2, z2])
    J_det = np.sin(phi)*(np.log(u)**2)/u
    a = make_state(r_sph1, 0, 1, 1/2, 1/2)
    b = make_state(r_sph1, 0, 1, 1/2, 1/2)
    return (J_det*np.dot(a, b.conj())).real
    #return J_det*(np.abs(make_state(r_sph1, 1, 1, 1/2, 1/2)[0])**2+np.abs(make_state(r_sph1, 1, 1, 1/2, 1/2)[1])**2)

def r12_sph(r1, theta1, phi1, r2, theta2, phi2):
    return np.sqrt(r1**2+r2**2-2*r1*r2*(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)-np.cos(theta1)*np.cos(theta2)))


n1, l1, j1, n2, l2, j2 = [0, 1, 1/2, 0, 1, 1/2]
n3, l3, j3, n4, l4, j4 = [0, 0, 1/2, 0, 0, 1/2]
J, M = [0, 0]

def norm_test(r_sph_doub):
    r1, theta1, phi1, r2, theta2, phi2 = r_sph_doub
    r_sph1 = np.array([r1, theta1, phi1])
    r_sph2 = np.array([r2, theta2, phi2])
    integrand = 0
    for m1 in np.arange(-j1, j1+1, 1):
        for m2 in np.arange(-j2, j2+1, 1):
            for m3 in np.arange(-j3, j3+1, 1):
                for m4 in np.arange(-j4, j4+1, 1):
                    if m1+m2!=M: continue
                    if m3+m4!=M: continue
                    a = make_state(r_sph1, n1, l1, j1, m1)
                    b = make_state(r_sph2, n2, l2, j2, m2)
                    c = make_state(r_sph1, n3, l3, j3, m3)
                    d = make_state(r_sph2, n4, l4, j4, m4)

                    integrand+=np.dot(a, c.conj())*np.dot(b, d.conj())*cgc(j1, j2, J, m1, m2, M)*cgc(j3, j4, J, m3, m4, M)
    
    denom = r12_sph(-np.log(r1), theta1, phi1, -np.log(r2), theta2, phi2)
    J_det = (np.sin(phi1)*(np.log(r1)**2)/r1)*(np.sin(phi2)*(np.log(r2)**2)/r2)
    #print(integrand*J_det/denom)
    return (integrand*J_det/denom).real

def sampler_three():
    while True:
        u = random.uniform(0, 1)
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, math.pi)
        yield (u, theta, phi)

def sampler_six():
    while True:
        u1 = random.uniform(0, 1)
        theta1 = random.uniform(0, 2*math.pi)
        phi1 = random.uniform(0, math.pi)
        u2 = random.uniform(0, 1)
        theta2 = random.uniform(0, 2*math.pi)
        phi2 = random.uniform(0, math.pi)
        yield (u1, theta1, phi1, u2, theta2, phi2)

domainsize = 4*math.pi**4
nmc = 100000

start = time.time()
#result, error = mcint.integrate(f, sampler_three(), measure=domainsize, n=nmc)
result, error = mcint.integrate(norm_test, sampler_six(), measure=domainsize, n=nmc)



print("Result = ", result.real)
print("Error estimate =", error)
print("Time =", time.time()-start)
