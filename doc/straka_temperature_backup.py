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


# Compute virtual temperature difference
print("*"*80)
print("We just convert the difference which is independent to Celsius or Kelvin")
_vt_kelvin = vt(z=z_c, t=-15.0)
print("Virtual temperature difference in C/K: "+str(_vt_kelvin))
print(exner(p_bar(z_c)))

print("*"*80)
print("Here, we to a conversion of the absolute temperature")
t = T_bar(z_c)
_vt_K_bar = vt(z=z_c, t=t)

t = C_to_K(K_to_C(T_bar(z_c)) - 15.0)
_vt_K_pert = vt(z=z_c, t=t)
_vt_K_diff = _vt_K_pert - _vt_K_bar
print("Virtual temperature difference in C/K: "+str(_vt_K_diff))

print("*"*80)
print("Should be (in C): "+str(-16.624))
print("")

#print(p_bar(z=z_c))
