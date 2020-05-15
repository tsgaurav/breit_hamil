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
from scipy.stats import expon

Sz = 0.5*np.array([[1, 0],[0, -1]])
Sx = 0.5*np.array([[0, 1],[1, 0]])
Sy = 0.5*np.array([[0, -1j],[1j, 0]])
alpha=1/137.035999084
g_e = 2.00231930436256

s_up = np.array([1, 0])
s_down = np.array([0, 1])

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
    eta = 2.0 * x / zeta
    return np.sqrt(2.0 * gamma(n+1) / (zeta * gamma(n+2*l+3)) ) * 2.0 * eta**l *\
              np.exp(-0.5*eta) * assoc_laguerre(eta, n, 2*l+2) / zeta


def lag_deriv(x, n, l):
    eta = 2.0 * x 
    t1 = 2*l*(eta)**(l-1)*np.exp(-eta*0.5)*assoc_laguerre(eta, n, 2*l+2)
    if n==0:
        t2 = 0
    else:
        t2 = 2*(eta)**l*np.exp(-eta*0.5)*assoc_laguerre(eta, n-1, 2*l+3)
    return (t1+t2)*np.sqrt(2.0 * gamma(n+1) / (gamma(n+2*l+3)) ) * 2.0  - laguerre_wave_function(x, n, l)


def dtheta_sphharm(m, n, theta, phi):
    return sph_harm(m, n, theta, phi)*m*1j

def dphi_sphharm(m, n, theta, phi):
    if m==n:
        return m*sph_harm(m, n, theta, phi)/np.tan(phi)
    else:
        return m*sph_harm(m, n, theta, phi)/np.tan(phi)+np.sqrt((n-m)*(n+m+1))*np.exp(-1j*theta)*sph_harm(m+1, n, theta, phi)


def LdotS_state(r_sph, n, l, j, mj):
    r, theta, phi = r_sph

    Stheta = -np.sin(theta)*Sx+np.cos(theta)*Sy
    Sphi = np.cos(phi)*np.cos(theta)*Sx + np.cos(phi)*np.sin(theta)*Sy - np.sin(phi)*Sz

    state_sum = 0.
    for ml in np.arange(-l, l+1, 1):

        theta_term = -r*dphi_sphharm(ml, l, theta, phi)*laguerre_wave_function(r, n, l)

        phi_term = r*dtheta_sphharm(ml, l, theta, phi)*laguerre_wave_function(r, n, l)

        state_sum += -1j*np.dot(theta_term*Stheta + phi_term*Sphi, s_up*cgc(l, 1/2, j, ml, 1/2, mj)+s_down*cgc(l, 1/2, j, ml, -1/2, mj))

    return state_sum
