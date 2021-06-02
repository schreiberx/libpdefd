import numpy as np
#import libpdefd


"""
Forward Euler
"""
def time_integrator_forward_euler(f, u, dt):
    return u + f(u)*dt

"""
4th order classic Runge Kutta
"""
def time_integrator_RK4(f, u, dt):
    
    k1 = f(u)
    
    k2 = f(u + k1*(dt*0.5))
    
    k3 = f(u + k2*(dt*0.5))
    
    k4 = f(u + k3*dt)
    
    # u + 1/6*dt * (k1 + 2*k2 + 2*k3 + k4)
    return u + (k1 + k2*2 + k3*2 + k4)*(1/6*dt)


def RK4(f, u, dt):
    return time_integrator_RK4(f, u, dt)



def time_integrator_leapfrog(f, u, dt, prev_u = None):
    if prev_u == None:
        return time_integrator_RK4(f, u, dt)

    else:
        return prev_u + f(u)*dt*2.0

"""
Matsuno time integrator

Matsuno, T. (1966). Numerical Integrations of the Primitive Difference Equations Method.
Journal of the Meteorological Society of Japan, 44(1), 76–84.

Eq. (3a) and (3b)

WARNING: Doesn't work that well for large time step sizes and rising bubble test case.
"""
def time_integrator_matsuno(f, u, dt):
    k1 = u + f(u)*dt
    return u + f(k1)*dt


"""
Dispatcher for time integration methods
"""
def time_integrator(name, f, u, dt):
    if name == "rk1":
        return time_integrator_forward_euler(f, u, dt)
    
    elif name == "rk4":
        return time_integrator_RK4(f, u, dt)
    
    elif name == "lf":
        return time_integrator_leapfrog(f, u, dt)
    
    elif name == "matsuno":
        return time_integrator_matsuno(f, u, dt)
    
    raise Exception("Unknown time integrator")



def gaussian_bump(
    mesh,
    ic_center,
    domain_size,
    boundary_condition,
    exp_parameter,
    x_scale_d3 = 1.0
):
    domain_size_min = np.min(domain_size)
    
    def initial_condition(x):
        t = (x - ic_center)/domain_size_min
        if len(t.shape) == 3:
            t[:,:,0] *= x_scale_d3
        
        t = t**2
        t = np.sum(t, axis=-1)
        return np.exp(-t*exp_parameter)
    
    
    if boundary_condition == "periodic":
        range_ = range(-10, 10+1)
    else:
        range_ = [0]
        
    num_dims = len(mesh.shape)-1
    
    import itertools
    range_ = itertools.product(*[range_ for _ in range(num_dims)])
        
    retval = np.zeros(mesh.shape[:-1])
    for rd in range_:
        retval += initial_condition(mesh + domain_size*np.array(rd))
    
    return retval
