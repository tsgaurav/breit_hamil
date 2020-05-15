import numpy as np
from scipy.special import gamma, assoc_laguerre, sph_harm
from sympy.physics.wigner import wigner_3j, wigner_6j
from sympy import N
import matplotlib.pyplot as plt
import mcint
import random
import math
import time
from scipy import integrate

start = time.time()

def thj(j1, j2, j3, m1, m2, m3):
    """
    3-j symbol
    ( j1 j2 j3 )
    ( m1 m2 m3 )
    """
    #return wigner3j(j1,j2,j3,m1,m2,m3)
    return N(wigner_3j(j1,j2,j3,m1,m2,m3))

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
    eta = 2.0 * x / zeta
    return np.sqrt(2.0 * gamma(n+1) / (zeta * gamma(n+2*l+3)) ) * 2.0 * eta**l *\
              np.exp(-0.5*eta) * assoc_laguerre(eta, n, 2*l+2) / zeta

def make_state(r_sph, n, l, j, mj):
    r, theta, phi = r_sph
    state_sum = np.array([0., 0.])
    for ml in np.arange(-l, l+1, 1):  #check limits
            state_sum= state_sum+cgc(l, 1/2, j, ml, 1/2, mj)*sph_harm(ml, l, theta, phi)*np.array([1., 0.])+cgc(l, 1/2, j, ml, -1/2, mj)*sph_harm(ml, l, theta, phi)*np.array([0., 1.])
    return laguerre_wave_function(r, n, l)*state_sum

def cart_to_sph(r_cart):
    x,y,z = r_cart
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    phi = np.arctan2(y,x)
    r_sph = np.array([r, theta, phi])
    return r_sph


def f(u, theta, phi):
    #u, theta, phi = r_sph1
    r_cart = np.array([u, theta, phi])
    r_sph1 = cart_to_sph(r_cart)
    #r_sph2 = cart_to_sph([x2, y2, z2])
    #J_det = u**2*np.sin(theta
    return (np.conj(make_state(r_sph1, 0, 0, 1/2, 1/2)[0])*make_state(r_sph1, 0, 0, 1/2, 1/2)[0] + np.conj(make_state(r_sph1, 0, 0, 1/2, 1/2)[1])*make_state(r_sph1, 0, 0, 1/2, 1/2)[1]) /np.sqrt(u**2+theta**2+phi**2)

result, error = integrate.nquad(f, [[-np.inf,np.inf],[-np.inf,np.inf],[-np.inf,np.inf]])

elapsed = time.time()-start
#r_sph1 = [1, 1.2, 0.9]
#print(np.conj(make_state(r_sph1, 0, 0, 1/2, 1/2)[0])*make_state(r_sph1, 1, 0, 1/2, 1/2)[0] + np.conj(make_state(r_sph1, 0, 0, 1/2, 1/2)[1])*make_state(r_sph1, 1, 0, 1/2, 1/2)[1])
print("Result =", result)
print("Error =", error)
print("Elasped time =", elapsed)

