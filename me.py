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


Sz = 0.5*np.array([[1, 0],[0, -1]])
Sx = 0.5*np.array([[0, 1],[1, 0]])
Sy = 0.5*np.array([[0, -1j],[1j, 0]])
alpha=1/137.035999084
g_e = 2.00231930436256


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

def laguerre_wave_function(x, n, l, zeta):
    """
    Laguerre function, see [A. E. McCoy and M. A. Caprio, J. Math. Phys. 57, (2016).] for details
    """
    eta = 2.0 * x / zeta
    return np.sqrt(2.0 * gamma(n+1) / (zeta * gamma(n+2*l+3)) ) * 2.0 * eta**l *\
              np.exp(-0.5*eta) * assoc_laguerre(eta, n, 2*l+2) / zeta

def make_state(r_sph, n, l, j, mj, zeta):
    r, theta, phi = r_sph
    state_sum = 0.
    for ml in np.arange(-l, l+1, 1):  #check limits
            state_sum= state_sum+np.array([cgc(l, 1/2, j, ml, 1/2, mj)*sph_harm(ml, l, theta, phi),cgc(l, 1/2, j, ml, -1/2, mj)*sph_harm(ml, l, theta, phi)])
    return laguerre_wave_function(r, n, l, zeta)*state_sum


def f(r_sph1):
    u, theta, phi = r_sph1
    #r_sph1 = np.array([u, theta, phi])
    #r_sph2 = cart_to_sph([x2, y2, z2])
    J_det = np.sin(phi)*(u**2)
    a = make_state(r_sph1, 0, 2, 3/2, 1/2)
    b = make_state(r_sph1, 0, 2, 3/2, 1/2)
    return (J_det*np.dot(a, b.conj())/(u))/gam.pdf(u, a_gam, scale=scale)
    #return J_det*(np.abs(make_state(r_sph1, 1, 1, 1/2, 1/2)[0])**2+np.abs(make_state(r_sph1, 1, 1, 1/2, 1/2)[1])**2)


def normal_ms(r_sph):
    zeta = 1
    r, theta, phi = r_sph
    J_det = np.sin(phi)*(r**2) 
    a_conj = make_state(r_sph, 2, 0,  1/2, 1/2, zeta).conj()
    b_pp_state = p_p_state(r_sph, 1, 0, 1/2, 1/2, zeta) 
    return J_det*np.dot(a_conj, b_pp_state)/2
"""
def spin_orb_1b(r_sph):
    r, theta, phi = r_sph
    J_det = np.sin(phi)*(r**2)
    a = make_state(r_sph, 0, 1, 3/2, 1/2)
    b = LdotS_state(r_sph, 0, 1, 3/2, 1/2)
    return 0.5*(alpha**2)*(J_det*np.dot(a, b.conj())/(r**3)).real
"""

def spin_orb_1b(r_sph):
    r, theta, phi = r_sph
    J_det = np.sin(phi)*(r**2)

    Stheta = -np.sin(theta)*Sx+np.cos(theta)*Sy
    Sphi = np.cos(phi)*np.cos(theta)*Sx + np.cos(phi)*np.sin(theta)*Sy - np.sin(phi)*Sz

    a_conj = make_state(r_sph, 0, 1, 3/2, 1/2).conj()
    b_phi = np.dot(Stheta, p_phi_state(r_sph, 0, 1, 3/2, 1/2))
    b_theta = np.dot(Sphi, p_theta_state(r_sph, 0, 1, 3/2, 1/2))

    integral = np.dot(a_conj, r*b_phi)-np.dot(a_conj, r*b_theta)
    return integral*J_det*(alpha**2)/(2*r**3)


def r12_sph(r1, theta1, phi1, r2, theta2, phi2):
    return np.sqrt(r1**2+r2**2-2*r1*r2*(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)-np.cos(theta1)*np.cos(theta2)))

def coul_2b(r_sph_doub, n1, l1, j1, n2, l2, j2, n3, l3, j3, n4, l4, j4, J, M, zeta):
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
                    a = make_state(r_sph1, n1, l1, j1, m1, zeta)
                    b = make_state(r_sph2, n2, l2, j2, m2, zeta)
                    c = make_state(r_sph1, n3, l3, j3, m3, zeta)
                    d = make_state(r_sph2, n4, l4, j4, m4, zeta)

                    integrand+=np.dot(a.conj(), c)*np.dot(b.conj(), d)*cgc(j1, j2, J, m1, m2, M)*cgc(j3, j4, J, m3, m4, M)
    
    denom = r12_sph(r1, theta1, phi1, r2, theta2, phi2)
    J_det = (np.sin(phi1)*((r1)**2))*(np.sin(phi2)*((r2)**2))
    #print(integrand*J_det/denom)
    return integrand*J_det/denom

def specific_ms(r_sph_doub, n1, l1, j1, n2, l2, j2, n3, l3, j3, n4, l4, j4, J, M, zeta):
    r1, theta1, phi1, r2, theta2, phi2 = r_sph_doub
    r_sph1 = np.array([r1, theta1, phi1])
    r_sph2 = np.array([r2, theta2, phi2])
    integrand = 0.
    for m1 in np.arange(-j1, j1+1, 1):
        for m2 in np.arange(-j2, j2+1, 1):
            for m3 in np.arange(-j3, j3+1, 1):
                for m4 in np.arange(-j4, j4+1, 1):
                    if m1+m2!=M: continue
                    if m3+m4!=M: continue
                    a = make_state(r_sph1, n1, l1, j1, m1, zeta)
                    b = make_state(r_sph2, n2, l2, j2, m2, zeta)
                    c_p_phi = p_phi_state(r_sph1, n3, l3, j3, m3, zeta)
                    c_p_theta = p_theta_state(r_sph1, n3, l3, j3, m3, zeta)
                    c_p_r = p_r_state(r_sph1, n3, l3, j3, m3, zeta)
                    d_p_phi = p_phi_state(r_sph2, n4, l4, j4, m4, zeta)
                    d_p_theta = p_theta_state(r_sph2, n4, l4, j4, m4, zeta)
                    d_p_r = p_r_state(r_sph2, n4, l4, j4, m4, zeta)

                    term1 = np.dot(a.conj(), c_p_r)*np.dot(b.conj(), d_p_r)
                    term2 = np.dot(a.conj(), c_p_phi)*np.dot(b.conj(), d_p_phi)
                    term3 = np.dot(a.conj(), c_p_theta)*np.dot(b.conj(), d_p_theta)

                    integrand+=(term1+term2+term3)*cgc(j1, j2, J, m1, m2, M)*cgc(j3, j4, J, m3, m4, M)

    J_det = (np.sin(phi1)*((r1)**2))*(np.sin(phi2)*((r2)**2))
    return integrand*J_det


def darwin(r_sph):
    r1, theta1, phi1= r_sph
    integrand = 0
    for m1 in np.arange(-j1, j1+1, 1):
        for m2 in np.arange(-j2, j2+1, 1):
            for m3 in np.arange(-j3, j3+1, 1):
                for m4 in np.arange(-j4, j4+1, 1):
                    if m1+m2!=M: continue
                    if m3+m4!=M: continue
                    a = make_state(r_sph, n1, l1, j1, m1)
                    b = make_state(r_sph, n2, l2, j2, m2)
                    c = make_state(r_sph, n3, l3, j3, m3)
                    d = make_state(r_sph, n4, l4, j4, m4)

                    integrand+=np.dot(a.conj(), c)*np.dot(b.conj(), d)*cgc(j1, j2, J, m1, m2, M)*cgc(j3, j4, J, m3, m4, M)

    J_det = (np.sin(phi1)*(r1**2))
    #print(integrand*J_det/denom)
    return g_e*np.pi*alpha**2*integrand*J_det/expon.pdf([r1, r2], scale=scale).prod()

def spin_spin(r_sph):
    r1, theta1, phi1= r_sph
    integrand = 0
    for m1 in np.arange(-j1, j1+1, 1):
        for m2 in np.arange(-j2, j2+1, 1):
            for m3 in np.arange(-j3, j3+1, 1):
                for m4 in np.arange(-j4, j4+1, 1):
                    if m1+m2!=M: continue
                    if m3+m4!=M: continue
                    a = make_state(r_sph, n1, l1, j1, m1)
                    b = make_state(r_sph, n2, l2, j2, m2)
                    c = np.dot(Sx+Sy+Sz, make_state(r_sph, n3, l3, j3, m3))
                    d = np.dot(Sx+Sy+Sz, make_state(r_sph, n4, l4, j4, m4))

                    integrand+=np.dot(a.conj(), c)*np.dot(b.conj(), d)*cgc(j1, j2, J, m1, m2, M)*cgc(j3, j4, J, m3, m4, M)

    J_det = (np.sin(phi1)*(r1**2))
    #print(integrand*J_det/denom)
    return g_e*8*np.pi*alpha**2*integrand*J_det/(3*expon.pdf([r1, r2], scale=scale).prod())

"""
def spin_orb_2b(r_sph_doub):

    r1, theta1, phi1, r2, theta2, phi2 = r_sph_doub
    r_sph1 = np.array([r1, theta1, phi1])
    r_sph2 = np.array([r2, theta2, phi2])

    Stheta1 = -np.sin(theta1)*Sx+np.cos(theta1)*Sy
    Sphi1 = np.cos(phi1)*np.cos(theta1)*Sx + np.cos(phi1)*np.sin(theta1)*Sy - np.sin(phi1)*Sz

    Stheta2 = -np.sin(theta2)*Sx+np.cos(theta2)*Sy
    Sphi2 = np.cos(phi2)*np.cos(theta2)*Sx + np.cos(phi2)*np.sin(theta2)*Sy - np.sin(phi2)*Sz

    integrand = 0
    for m1 in np.arange(-j1, j1+1, 1):
        for m2 in np.arange(-j2, j2+1, 1):
            for m3 in np.arange(-j3, j3+1, 1):
                for m4 in np.arange(-j4, j4+1, 1):
                    if m1+m2!=M: continue
                    if m3+m4!=M: continue

                    a_conj = make_state(r_sph1, n1, l1, j1, m1).conj()
                    b_conj = make_state(r_sph2, n2, l2, j2, m2).conj()
                    d = make_state(r_sph2, n4, l4, j4, m4)
                    c_p_phi = p_phi_state(r_sph1, n3, l3, j3, m3)
                    c_p_theta = p_theta_state(r_sph1, n3, l3, j3, m3)

                    term1 = np.dot(a_conj, np.dot(Stheta1, r1*c_p_phi))*np.dot(b_conj, d)*-2
                    term2 = np.dot(a_conj, np.dot(Sphi1, r1*c_p_theta))*np.dot(b_conj, d)*2
                    term3 = np.dot(a_conj,r1*c_p_phi)*np.dot(b_conj, np.dot(Stheta2, d))*-1
                    term4 = np.dot(a_conj,r1*c_p_theta)*np.dot(b_conj, np.dot(Sphi2, d))
                    term5 = np.dot(a_conj, np.dot(Stheta1, c_p_phi))*np.dot(b_conj, r2*d)*2
                    term6 = np.dot(a_conj, np.dot(Sphi1, c_p_theta))*np.dot(b_conj, r2*d)*-2
                    term7 = np.dot(a_conj,c_p_phi)*np.dot(b_conj, np.dot(r2*Stheta2, d))
                    term8 = np.dot(a_conj,c_p_theta)*np.dot(b_conj, np.dot(r2*Sphi2, d))*-1

                    integrand+=(term1+term2+term3+term4+term5+term6+term7+term8)*cgc(j1, j2, J, m1, m2, M)*cgc(j3, j4, J, m3, m4, M)

    J_det = (np.sin(phi1)*(r1**2))*(np.sin(phi2)*(r2**2))
    return alpha**2*J_det*integrand/(4*r12_sph(r1, theta1, phi1, r2, theta2, phi2)**3)
"""
def spin_orb_2b(r_sph_doub, n1, l1, j1, n2, l2, j2, n3, l3, j3, n4, l4, j4, J, M):
    r1, theta1, phi1, r2, theta2, phi2 = r_sph_doub
    r_sph1 = np.array([r1, theta1, phi1])
    r_sph2 = np.array([r2, theta2, phi2])
    
    Stheta1 = -np.sin(theta1)*Sx+np.cos(theta1)*Sy
    Sphi1 = np.cos(phi1)*np.cos(theta1)*Sx + np.cos(phi1)*np.sin(theta1)*Sy - np.sin(phi1)*Sz
    
    J_det = (np.sin(phi1)*(r1**2))*(np.sin(phi2)*(r2**2))
    integrand = 0.
    
    for m1 in np.arange(-j1, j1+1, 1):
        for m2 in np.arange(-j2, j2+1, 1):
            for m3 in np.arange(-j3, j3+1, 1):
                for m4 in np.arange(-j4, j4+1, 1):
                    if m1+m2!=M: continue
                    if m3+m4!=M: continue
                        
                    a_conj = make_state(r_sph1, n1, l1, j1, m1).conj()
                    b_conj = make_state(r_sph2, n2, l2, j2, m2).conj()
                    c_p_phi = np.dot(Stheta1, p_phi_state(r_sph1, n3, l3, j3, m3))
                    c_p_theta = np.dot(Sphi1, p_theta_state(r_sph1, n3, l3, j3, m3))
                    c_Sphi = np.dot(Sphi1, make_state(r_sph1, n3, l3, j3, m3))
                    c_Stheta = np.dot(Stheta1, make_state(r_sph1, n3, l3, j3, m3))
                    d_p_phi = p_phi_state(r_sph2, n4, l4, j4, m4)
                    d_p_theta = p_theta_state(r_sph2, n4, l4, j4, m4)
                    d = make_state(r_sph2, n4, l4, j4, m4)
                    
                    term1 = 2*np.dot(a_conj, r1*c_Sphi)*np.dot(b_conj, d_p_theta)
                    term2 = -2*np.dot(a_conj, r1*c_Stheta)*np.dot(b_conj, d_p_phi)
                    term3 = -2*np.dot(a_conj, c_Sphi)*np.dot(b_conj, r2*d_p_theta)
                    term4 = 2*np.dot(a_conj, c_Stheta)*np.dot(b_conj, r2*d_p_phi)
                    term5 = -np.dot(a_conj, r1*c_p_theta)*np.dot(b_conj, d)
                    term6 = np.dot(a_conj, r1*c_p_phi)*np.dot(b_conj, d)
                    term7 = np.dot(a_conj, c_p_theta)*np.dot(b_conj, r2*d)
                    term8 = -np.dot(a_conj, c_p_phi)*np.dot(b_conj, r2*d)
                    
                    integrand+=(term1+term2+term3+term4+term5+term6+term7+term8)*cgc(j1, j2, J, m1, m2, M)*cgc(j3, j4, J, m3, m4, M)
    return J_det*integrand/(r12_sph(r1, theta1, phi1, r2, theta2, phi2)**3)*(alpha**2/4)

def orb_orb_2b(r_sph_doub):
    
    r1, theta1, phi1, r2, theta2, phi2 = r_sph_doub
    r_sph1 = np.array([r1, theta1, phi1])
    r_sph2 = np.array([r2, theta2, phi2])
    
    denom = r12_sph(r1, theta1, phi1, r2, theta2, phi2)
    
    integrand = 0
    for m1 in np.arange(-j1, j1+1, 1):
        for m2 in np.arange(-j2, j2+1, 1):
            for m3 in np.arange(-j3, j3+1, 1):
                for m4 in np.arange(-j4, j4+1, 1):
                    if m1+m2!=M: continue
                    if m3+m4!=M: continue
    
                    a_conj = make_state(r_sph1, n1, l1, j1, m1).conj()
                    b_conj = make_state(r_sph2, n2, l2, j2, m2).conj()
                    c_p_phi = p_phi_state(r_sph1, n3, l3, j3, m3)
                    c_p_theta = p_theta_state(r_sph1, n3, l3, j3, m3)
                    c_p_r = p_r_state(r_sph1, n3, l3, j3, m3)
                    d_p_phi = p_phi_state(r_sph2, n4, l4, j4, m4)
                    d_p_theta = p_theta_state(r_sph2, n4, l4, j4, m4)
                    d_p_r = p_r_state(r_sph2, n4, l4, j4, m4)
                    
                    term1 = np.dot(a_conj, c_p_r)*np.dot(b_conj, d_p_r)
                    term2 = np.dot(a_conj, c_p_theta)*np.dot(b_conj, d_p_theta)
                    term3 = np.dot(a_conj, c_p_phi)*np.dot(b_conj, d_p_phi)
                    term4 = np.dot(a_conj, r1*r1*c_p_r)*np.dot(b_conj, d_p_r)/denom**2
                    term5 = -2*np.dot(a_conj, r1*c_p_r)*np.dot(b_conj, r2*d_p_r)/denom**2
                    term6 = np.dot(a_conj, c_p_r)*np.dot(b_conj, r2*r2*d_p_r)/denom**2
                    
                    integrand+=(term1+term2+term3+term4+term5+term6)*cgc(j1, j2, J, m1, m2, M)*cgc(j3, j4, J, m3, m4, M)
    
    J_det = (np.sin(phi1)*(r1**2))*(np.sin(phi2)*(r2**2))
    return -alpha**2*J_det*integrand/(2*denom)

