


class _Boundary_Base:
    def __init__(self, type):
        self.type = type
        
    def __str__(self):
        retstr = ""
        retstr += self.type
        return retstr


class BoundaryPeriodic(_Boundary_Base):
    def __init__(self):
        _Boundary_Base.__init__(self, "periodic")

    def __str__(self):
        retstr = ""
        retstr += self.type
        return retstr


class BoundaryDirichlet(_Boundary_Base):
    def __init__(self, dirichlet_value):
        _Boundary_Base.__init__(self, "dirichlet")
        self.dirichlet_value = dirichlet_value

    def __str__(self):
        retstr = ""
        retstr += self.type+"("+str(self.dirichlet_value)+")"
        return retstr


class BoundaryNeumannExtrapolated(_Boundary_Base):
    """
    Neumann boundary without DoF on Boundary
    
    This BC relates to reconstruct a Neumann DoF of 0th order
    which matches the required derivative.
    
    The difference to the previous boundary condition is that
    there *exists no DoF at the boundary*. 
    """
    def __init__(self, neumann_value = 0, diff_order = 1):
        _Boundary_Base.__init__(self, "neumann_extrapolated")
        self.neumann_value = neumann_value
        
        if diff_order <= 0:
            raise Exception("Neumann derivative only for >0 order")
        
        self.neumann_diff_order = diff_order

    def __str__(self):
        retstr = ""
        retstr += self.type+"("+str(self.neumann_diff_order)+")"
        return retstr



class BoundarySymmetric(_Boundary_Base):
    """
    Make the boundary symmetric which means that we have Neumann df/dx*n=0 boundary conditions
    
    This BC relates to a DoF on the boundaries at which the
    given order and value is fulfilled.
    
    The difference to the previous boundary condition is that
    there exists a DoF at this boundary. 
    """
    def __init__(self, flip_sign = False):
        _Boundary_Base.__init__(self, "symmetric")
        self.flip_sign = flip_sign
        
    def __str__(self):
        retstr = ""
        retstr += self.type
        return retstr

