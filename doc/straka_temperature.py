#! /usr/bin/env python3

"""
Reproducer of minimum virtual temperature at t=0
"""

z_c = 3.0e3

g = 9.81
R_d = 287
C_p = 1004
T_s = 300
p_0 = 100e3



def T_bar(z):
    return T_s - z*g/C_p

def p_bar(z):
    # "Wrong" in paper
    #return p_0 * (T_bar(z) / T_s)**(R_d / C_p)
    return p_0 * (T_bar(z) / T_s)**(C_p / R_d)

def C_to_K(t):
    return t + 273.15

def K_to_C(t):
    return t - 273.15

def exner(p):
    return (p/p_0)**(R_d/C_p)



# Compute virtual temperature
delta_t = C_to_K(-15.0)

def vt(z, t):
    return t/exner(p_bar(z))


print("*"*80)
_vt = vt(z=z_c, t=-15.0)
print(_vt)

print("*"*80)
_vt_celsius = K_to_C(_vt)
print(_vt_celsius)

