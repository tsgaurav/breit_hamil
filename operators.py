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

def laguerre_wave_function(x, n, l, zeta):
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
        t2 = -2*(eta)**l*np.exp(-eta*0.5)*assoc_laguerre(eta, n-1, 2*l+3)
    return (t1+t2)*np.sqrt(2.0 * gamma(n+1) / (gamma(n+2*l+3)) ) * 2.0  - laguerre_wave_function(x, n, l)

def lag_deriv(x, n, l, zeta):
    eta = 2.0 * x /zeta
    t1 = (2/zeta)*l*(eta)**(l-1)*np.exp(-eta*0.5)*assoc_laguerre(eta, n, 2*l+2)
    if n==0:
        t2 = 0
    else:
        t2 = -(2/zeta)*(eta)**l*np.exp(-eta*0.5)*assoc_laguerre(eta, n-1, 2*l+3)
    return (t1+t2)*np.sqrt(2.0 * gamma(n+1) / (gamma(n+2*l+3)) ) * 2.0  - laguerre_wave_function(x, n, l, zeta)

def lag_double_deriv(x, n, l, zeta):
    eta = 2.0 * x / zeta
    t1 = (4/zeta**2)*l*(l-1)*eta**(l-2)*np.exp(-eta*0.5)*assoc_laguerre(eta, n-1, 2*l+2)
    if n==0:
        t2 = 0
        t3 = 0
    elif n==1:
        t2 = -2*(8*l*eta**(l-2)/zeta**2 - 2*eta**l / zeta)*np.exp(-eta*0.5)*assoc_laguerre(eta, n-1, 2*l+3)
        t3 = 0
    else:
        t2 = -2*(8*l*eta**(l-2)/zeta**2 - 2*eta**l / zeta)*np.exp(-eta*0.5)*assoc_laguerre(eta, n-1, 2*l+3)
        t3 = 4*(eta**l)*np.exp(-eta*0.5)*assoc_laguerre(eta, n-2, 2*l+4)/zeta**2
    return (t1+t2+t3)*np.sqrt(2.0 * gamma(n+1) / (gamma(n+2*l+3)) ) * 2.0 - 2*l* laguerre_wave_function(x, n, l, zeta) - lag_deriv(x, n, l, zeta)


def dtheta_sphharm(m, n, theta, phi):
    return sph_harm(m, n, theta, phi)*m*1j

def dphi_sphharm(m, n, theta, phi):
    if m==n:
        return m*sph_harm(m, n, theta, phi)/np.tan(phi)
    else:
        return m*sph_harm(m, n, theta, phi)/np.tan(phi)+((n-m)*(n+m+1))**0.5*np.exp(-1j*theta)*sph_harm(m+1, n, theta, phi)

def dtheta2_sph_harm(m, n, theta, phi):
    return -sph_harm(m, n, theta, phi)*m**2

def dphi2_sph_harm(m, n, theta, phi):
    if m==n:
        return m*(m/np.tan(phi)**2 - 1/np.sin(phi)**2)*sph_harm(m, n, theta, phi)
    elif (n-m)==1:
        return m*(m/np.tan(phi)**2 - 1/np.sin(phi)**2)*sph_harm(m, n, theta, phi) + \
               np.sqrt((n-m)*(n+m+1))*np.exp(-1j*theta)*sph_harm(m+1, n, theta, phi)*(2*m+1)/np.tan(phi)
    else:
        return m*(m/np.tan(phi)**2 - 1/np.sin(phi)**2)*sph_harm(m, n, theta, phi) + \
               np.sqrt((n-m)*(n+m+1))*np.exp(-1j*theta)*sph_harm(m+1, n, theta, phi)*(2*m+1)/np.tan(phi)+ \
               np.sqrt((n-m)*(n-m-1)*(m+n+2)*(m+n+1)) * np.exp(-2j*theta)*sph_harm(m+2, n, theta, phi)


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


def p_p_state(r_sph, n, l, j, mj, zeta):
    r, theta, phi = r_sph
    state_sum = 0.
    for ml in np.arange(-l, l+1, 1):

        r_term  =  (2*lag_deriv(r, n, l, zeta)/r+lag_double_deriv(r, n, l, zeta))*sph_harm(ml, l, theta, phi)

        theta_term = laguerre_wave_function(r, n, l, zeta) * dtheta2_sph_harm(ml, l, theta, phi)/(r*r*np.sin(phi)**2)
        
        phi_term = laguerre_wave_function(r, n, l, zeta) * (dphi2_sph_harm(ml, l, theta, phi) + dphi_sphharm(ml, l, theta, phi)/np.tan(phi))/r**2 

        state_sum += (s_up*cgc(l, 1/2, j, ml, 1/2, mj)+s_down*cgc(l, 1/2, j, ml, -1/2, mj)) * (r_term + theta_term + phi_term)

    return -state_sum

"""
def p_phi_state(r_sph, n, l, j, mj):
    r, theta, phi = r_sph
    state_sum = 0.

    for ml in np.arange(-l, l+1, 1):
        phi_term = dphi_sphharm(ml, l, theta, phi)*laguerre_wave_function(r, n, l)/r
        state_sum+=phi_term * (s_up*cgc(l, 1/2, j, ml, 1/2, mj)+s_down*cgc(l, 1/2, j, ml, -1/2, mj))
    return -1j*state_sum

def p_theta_state(r_sph, n, l, j, mj):
    r, theta, phi = r_sph
    state_sum = 0.

    for ml in np.arange(-l, l+1, 1):
        theta_term = dtheta_sphharm(ml, l, theta, phi)*laguerre_wave_function(r, n, l)/(r*np.sin(phi))
        state_sum+=theta_term * (s_up*cgc(l, 1/2, j, ml, 1/2, mj)+s_down*cgc(l, 1/2, j, ml, -1/2, mj))
    return -1j*state_sum

def p_r_state(r_sph, n, l, j, mj):
    r, theta, phi = r_sph
    state_sum = 0.
    for ml in np.arange(-l, l+1, 1):
        r_term = lag_deriv(r, n, l)*sph_harm(ml, l, theta, phi)
        state_sum+=r_term*(s_up*cgc(l, 1/2, j, ml, 1/2, mj)+s_down*cgc(l, 1/2, j, ml, -1/2, mj))
    return -1j*state_sum

"""


def p_r_state(r_sph, n, l, j, mj, zeta):
    r, theta, phi = r_sph
    state_sum = 0.
    radial = lag_deriv(r, n, l, zeta)
    for ml in np.arange(-l, l+1, 1):
        r_term = sph_harm(ml, l, theta, phi)
        state_sum+=r_term*(s_up*cgc(l, 1/2, j, ml, 1/2, mj)+s_down*cgc(l, 1/2, j, ml, -1/2, mj))
    return -1j*state_sum*radial

def p_theta_state(r_sph, n, l, j, mj, zeta):
    r, theta, phi = r_sph
    state_sum = 0.
    radial = laguerre_wave_function(r, n, l, zeta)/(r*np.sin(phi))
    for ml in np.arange(-l, l+1, 1):
        theta_term = dtheta_sphharm(ml, l, theta, phi)
        state_sum+=theta_term * (s_up*cgc(l, 1/2, j, ml, 1/2, mj)+s_down*cgc(l, 1/2, j, ml, -1/2, mj))
    return -1j*state_sum*radial

def p_phi_state(r_sph, n, l, j, mj, zeta):
    r, theta, phi = r_sph
    state_sum = 0.
    radial = laguerre_wave_function(r, n, l, zeta)/r
    for ml in np.arange(-l, l+1, 1):
        phi_term = dphi_sphharm(ml, l, theta, phi)
        state_sum+=phi_term * (s_up*cgc(l, 1/2, j, ml, 1/2, mj)+s_down*cgc(l, 1/2, j, ml, -1/2, mj))
    return -1j*state_sum*radial
