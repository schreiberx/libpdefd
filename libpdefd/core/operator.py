from libpdefd.core.gridinfo import *
import libpdefd.core.variable as variable
import libpdefd.core.fd_weights_explicit as fdwe

import libpdefd.matrix_vector_array.libpdefd_matrix_setup as matrix_setup
import libpdefd.matrix_vector_array.libpdefd_matrix_compute as matrix_compute
import libpdefd.matrix_vector_array.libpdefd_vector_array as libpdefd_vector_array



_default_params = {
    'operator_diff__min_approx_order': 2
}


def set_default_param(name, value):
    if not name in _default_params:
        raise Exception("Parameter '"+name+"' not in default params")
    
    _default_params[name] = value



class _operator_base:
    def __init__(self):
        self._L_sparse_compute = None
        self._L_sparse_setup = None
        self._name = "TMP"
        self._baked = False
    
    
    def setup(self, name):
        self._name = name
    
    
    def bake(self):
        """
        Bake setup matrix to one which is computationally efficient
        """
        self._L_sparse_compute = matrix_compute.matrix_sparse(self._L_sparse_setup)
        self._L_sparse_setup = None 
        self._baked = True
        return self



    """
    def get_L(self):
        return self._L_sparse_setup.toarray()
    
    def get_c(self):
        return self._c
    
    def get_L_sparse_setup(self):
        return self._L_sparse_setup
    
    def solve(self, x, solver_tol=None, max_iterations=None):
        raise Exception("TODO")
        
        if isinstance(x, np.ndarray):
            import libtide.linalg.iterative_solvers as iterative_solvers
            b = x.flatten()-self._c
            
            retval = iterative_solvers.solve(self._L_sparse_setup, b, solver="gmres", solver_tol=solver_tol, max_iterations=max_iterations)
            
            return retval.reshape(self._dst_grid.shape)
            #return self._L_sparse_setup.dot(x) + self._c
        
        elif isinstance(x, VariableND) or isinstance(x, variable._VariableND_Base):
            raise Exception("TODO")
            assert self._L_sparse_setup.shape[0] == self._c.shape[0]
            
            data = (self._L_sparse_setup.dot(x._data.flatten()) + self._c).reshape(self._dst_grid.shape)
            
            return VariableND(
                self._dst_grid,
                name = self._name+"("+x.name+")",
                data = data
            )
        
        raise Exception("Don't know how to handle '"+str(type(x))+"'")
    """
    
    def apply(self, x):
        """
        Make sure that everything got "baked", hence the compuational efficient matrix is now avaialble
        """
        if self._L_sparse_compute == None:
            raise Exception("The layout optimized for computations is not yet set up. Please call bake() on the operator to do so!")
        
        if isinstance(x, variable._VariableND_Base):
            assert self._L_sparse_compute.shape[0] == self._c.shape[0]
            
            data = self._L_sparse_compute.dot_add_reshape(x._data, self._c, self._dst_grid.shape)
            return variable._VariableND_Base(data)
        
        elif isinstance(x, libpdefd_vector_array._vector_array_base):
            assert self._L_sparse_compute.shape[0] == self._c.shape[0]
            
            data = self._L_sparse_compute.dot_add_reshape(x, self._c, self._dst_grid.shape)
            return libpdefd_vector_array._vector_array_base(data = data)
        
        elif isinstance(x, np.ndarray):
            assert len(x.shape) == 1
            assert x.shape[0] == self._L_sparse_compute.shape[0]
            return self._L_sparse_compute.dot_add(x, self._c)
        
        raise Exception("Don't know how to handle '"+str(type(x))+"'")
    
    
    def __call__(self, x):
        return self.apply(x)
    
    def __add__(self, a):
        retval = self.__class__()
        retval._L_sparse_setup = self._L_sparse_setup + a._L_sparse_setup
        return retval

    def __sub__(self, a):
        retval = self.__class__()
        retval._L_sparse_setup = self._L_sparse_setup - a._L_sparse_setup
        return retval

    def __mul__(self, a):
        retval = self.__class__()
        if isinstance(a, VariableND):
            retval._L_sparse_setup = self._L_sparse_setup * a._L_sparse_setup
        else:
            retval._L_sparse_setup = self._L_sparse_setup * a
        return retval

    def __rmul__(self, a):
        retval = self.__class__()
        if isinstance(a, VariableND):
            retval._L_sparse_setup = a._L_sparse_setup * self._L_sparse_setup
        else:
            retval._L_sparse_setup = a * self._L_sparse_setup
        return retval

    def __div__(self, a):
        retval = self.__class__()
        retval._L_sparse_setup = self._L_sparse_setup / a._L_sparse_setup
        return retval

    def __iadd__(self, a):
        self._L_sparse_setup += a._L_sparse_setup
        return self

    def __isub__(self, a):
        self._L_sparse_setup -= a._L_sparse_setup
        return self

    def __imul__(self, a):
        self._L_sparse_setup *= a._L_sparse_setup
        return self

    def __idiv__(self, a):
        self._L_sparse_setup /= a._L_sparse_setup
        return self

    def __neg__(self):
        retval = self.__class__()
        retval._L_sparse_setup = -self._L_sparse_setup
        return retval

    def __pos__(self):
        retval = self.__class__()
        retval._L_sparse_setup = self._L_sparse_setup
        return retval

    def __str__(self):
        retstr = ""
        retstr += "shape: "+str(self._L_sparse_setup.shape)
        return retstr
    
    def to_numpy_array(self):
        if self._baked:
            return self._L_sparse_compute.to_numpy_array()
        else:
            return self._L_sparse_setup.to_numpy_array()



class OperatorDiff1D(_operator_base):
    """
    OperatorDiff1Derential operator class
    """
    def __init__(self, *args, **kwargs):
        _operator_base.__init__(self)

        if len(args) > 0 or len(kwargs) > 0:
            self.setup(*args, **kwargs)
    
    
    def _get_src_info(
        self,
        dst_x : float,      # coordinate on which to compute the finite differences
        diff_order : int,
        min_approx_order : int,
        src_idx_closest = 0,    # Closest index in source coordinates (used for optimization)
        main_dof_execution = True
    ):
        if 0:
            print("="*80)
            print("_get_src_info(...)")
            print("="*80)
            
            print(" + src_grid.x_stencil_dofs: "+str(self._src_grid.x_stencil_dofs))
            print(" + src_grid.x_dofs: "+str(self._src_grid.x_dofs))
            print(" + dst_grid.x_stencil_dofs: "+str(self._dst_grid.x_stencil_dofs))
            print(" + dst_grid.x_dofs: "+str(self._dst_grid.x_dofs))
            print(" + dst_x: "+str(dst_x))
            
            print("="*80)
        
        """
        Determine src coordinates and indices to compute stencil for point dst_x on destination grid
        
        diff_order: Order of differential operator
        min_approx_order: Approximation order (accuracy)
        
        Return: (src_range, src_x_dofs, src_idx_closest)
            src_range: range in linear operator
            src_x_dofs: DoFs to compute the differential operator on
            src_idx_closest: Index to speed up things
        """
        
        dst_x_domain_periodic_shifted = False
        
        if self._src_grid.boundaries[0].type == "periodic":
            """
            First, we cope with the special situation of dst_x being the first grid point on a staggered grid
            
            We then need to shift dst_x
            
            Domain:
            |---------------------------|
            
            src_grid:
                |------|------|------|------|
            
            dst_grid:
            |------|------|------|------|
            |
            V
      Special case
            |
            -----------------------------------|
                                               V
                                               |
                                    Converted to this point
                        This will include the last DoF for creating the stencil
            
            
            src_x_dofs:
                |------|------|------| 
            
            dst_x_dofs:
            |------|------|------|
            """
            if dst_x >= self._src_grid.domain_start and dst_x <= self._src_grid.x_dofs[0]:
                dst_x += self._src_grid.domain_size
                dst_x_domain_periodic_shifted = True
            
            """
            Finally, we need to fix a potentially cached src_idx_closest which
            is not correct anymore
            """
            if src_idx_closest >= self._src_grid.num_stencil_grid_points-2:
                src_idx_closest = 0
        
        
        """
        Given the current point on the dst variable,
        search for closest point on src variable
        """
        src_idx_closest_dist = np.inf
        for src_idx in range(src_idx_closest, self._src_grid.num_stencil_grid_points):
            """
            We use x_stencil_dofs here since this also includes the
            nodes for the Neumann and Dirichlet BC which will be
            "extracted" handled later on. This strongly simplifies things. 
            """
            
            src_x = self._src_grid.x_stencil_dofs[src_idx]
            
            """
                                src_x
                                 |
                                 V
            |------|------|------|------|------|
            ----|------|------|------|------|------|
                *      *      *
            
            * = potential candidates

            Filter out those which are already beyond the current one (+eps)
            We need this for the correct staggered indexing.
            """
            if src_x > dst_x + self.aligned_eps:
                break
            
            dist = abs(src_x - dst_x)
            if dist < src_idx_closest_dist:
                src_idx_closest = src_idx
                src_idx_closest_dist = dist
        
        assert src_idx_closest != None, "Internal error"
        
        """
        GRID POINT ALIGNMENT
        
        Check if the grid points on src/dst grid are both aligned
        """
        if main_dof_execution:
            if self._src_grid is self._dst_grid:
                assert self._src_grid.x_stencil_dofs[src_idx_closest] == dst_x, "There seems to be a problem with the coordinates"
        
        aligned_grid_point = (src_idx_closest_dist <= self.aligned_eps)
        
        if 0:
            print("*"*80)
            print(" + src_grid.x_stencil_dofs: "+str(self._src_grid.x_stencil_dofs))
            print(" + dst_grid.x_stencil_dofs: "+str(self._dst_grid.x_stencil_dofs))
            print(" + src_idx_closest: "+str(src_idx_closest))
            print(" + src_idx_closest_dist: "+str(src_idx_closest_dist))
            print(" + dst_x: "+str(dst_x))
            print(" + aligned_grid_point: "+str(aligned_grid_point))
            #sys.exit(1)
        
        """
        Determine number of required DoFs
        
        We only use symmetric stencils except at some boundaries
        """
        
        if aligned_grid_point:
            """
            Aligned grid point
            """
            
            num_stencil_dofs = min_approx_order + diff_order
            
            if num_stencil_dofs % 2 == 0:
                num_stencil_dofs += 1
            
            real_approx_order_equispaced = num_stencil_dofs - diff_order
            
            real_approx_order_equispaced_postprocessed = False

            if min_approx_order == 1:
                if not real_approx_order_equispaced_postprocessed:
                    if self._src_grid.boundaries[0].type in ["dirichlet", "neumann_extrapolated"] and self._dst_grid.boundaries[0].type in ["symmetric"]:
                        #if self._src_grid.x_stencil_dofs[0] > self._dst_grid.x_stencil_dofs[0]:
                        if True:
                            real_approx_order_equispaced = max(1, real_approx_order_equispaced-1)
                            real_approx_order_equispaced_postprocessed = True
                    
                if not real_approx_order_equispaced_postprocessed: 
                    if self._src_grid.boundaries[1].type in ["dirichlet", "neumann_extrapolated"] and self._dst_grid.boundaries[1].type in ["symmetric"]:
                        #if self._src_grid.x_stencil_dofs[-1] < self._dst_grid.x_stencil_dofs[-1]:
                        if True:
                            real_approx_order_equispaced = max(1, real_approx_order_equispaced-1)
                            real_approx_order_equispaced_postprocessed = True
            
            if not real_approx_order_equispaced_postprocessed:
                
                if self._src_grid.boundaries[0].type in ["periodic", "symmetric"] and self._src_grid.boundaries[1].type in ["periodic", "symmetric"]:
                #if self._src_grid.boundaries[0].type in ["periodic"] and self._src_grid.boundaries[1].type in ["periodic"]:
                    if diff_order % 2 == 0:
                        real_approx_order_equispaced += 1
                
                else:
                    """
                    TODO: Figure out why that's the case 
                    """
                    if self._src_grid.staggered and self._dst_grid.staggered:
                        pass
                    
                    
                    elif diff_order <= 3:
                    #if diff_order <= 3 and min_approx_order == 1:
                        real_approx_order_equispaced = max(2, real_approx_order_equispaced)
                    
                    """
                    if self._src_grid.boundaries[0].type in ["dirichlet0"] and self._src_grid.boundaries[1].type in ["dirichlet0"]:
                        
                    else:
                        if diff_order <= 3:
                        #if diff_order <= 3 and min_approx_order == 1:
                            real_approx_order_equispaced = max(2, real_approx_order_equispaced)
                    """
            
            num_stencil_default_offset = -(num_stencil_dofs//2)
            
            """
            Example: For 2nd order method and 1st order derivative, 3 stencil points are sufficient
            """
            #if approx_order == 2 and diff_order == 1:
            #    assert num_stencil_dofs == 3
        
        else:
            """
            Non-aligned grid point
            """
            
            """
            Update the dst dof field with the information that there's
            at least one grid point which is not aligned.
            """
            self.all_dst_dof_points_aligned = False
            
            num_stencil_dofs = min_approx_order + diff_order
            
            # Make number even
            if num_stencil_dofs % 2 == 1:
                num_stencil_dofs += 1
            
            real_approx_order_equispaced = num_stencil_dofs - diff_order
            
            real_approx_order_equispaced_postprocessed = False
            
            if min_approx_order == 1:
                if not real_approx_order_equispaced_postprocessed: 
                    if self._src_grid.boundaries[0].type in ["dirichlet", "neumann_extrapolated"] and self._dst_grid.boundaries[0].type in ["symmetric"]:
                        if self._src_grid.x_stencil_dofs[0] > self._dst_grid.x_stencil_dofs[0]:
                            real_approx_order_equispaced = max(1, real_approx_order_equispaced-1)
                            real_approx_order_equispaced_postprocessed = True
                    
                if not real_approx_order_equispaced_postprocessed: 
                    if self._src_grid.boundaries[1].type in ["dirichlet", "neumann_extrapolated"] and self._dst_grid.boundaries[1].type in ["symmetric"]:
                        if self._src_grid.x_stencil_dofs[-1] < self._dst_grid.x_stencil_dofs[-1]:
                            real_approx_order_equispaced = max(1, real_approx_order_equispaced-1)
                            real_approx_order_equispaced_postprocessed = True
            
            if not real_approx_order_equispaced_postprocessed:
                if self._src_grid.boundaries[0].type in ["periodic", "symmetric"] and self._src_grid.boundaries[1].type in ["periodic", "symmetric"]:
                #if src_grid.boundaries[0].type == "periodic" and src_grid.boundaries[0].type == "periodic":
                
                    if diff_order % 2 == 1:
                        real_approx_order_equispaced += 1
                    
                else:
                    """
                    TODO: Figure out why that's the case 
                    """
                    if diff_order < 3:
                        real_approx_order_equispaced = max(2, real_approx_order_equispaced)
            
            num_stencil_default_offset = -(num_stencil_dofs//2)+1
            
            """
            Example: For 2nd order method and 1st order derivative, 2 stencil points are sufficient
            """
            if min_approx_order == 2 and diff_order == 1:
                assert num_stencil_dofs == 4
        
        """
        Store real approximation order
        """
        if main_dof_execution:
            if self.real_approx_order_equispaced is None:
                self.real_approx_order_equispaced = real_approx_order_equispaced
                
            else:
                """
                Use max instead of min, since there are some special cases where
                """
                self.real_approx_order_equispaced = min(self.real_approx_order_equispaced, real_approx_order_equispaced)
                assert self.real_approx_order_equispaced >= min_approx_order
        
        
        """
        Safety check
        
        Make sure that there are enough grid point to avoid overlapping stencils of boundaries operators
        """
        if 1:
            assert self._src_grid.num_dofs > num_stencil_dofs
            assert self._dst_grid.num_dofs > num_stencil_dofs
        
        """
        Determine concrete coordinates on source grid on which to compute the stencil on
        """
        if aligned_grid_point and diff_order == 0:
            """
            Special case of diff_order=0 (interpolation):
            If the grids are perfectly aligned, there's only one point we need
            """
            src_range = np.array([src_idx_closest])
        else:
            src_range = np.arange(num_stencil_default_offset, num_stencil_default_offset+num_stencil_dofs) + src_idx_closest

        if not aligned_grid_point:
            if dst_x < self._src_grid.x_stencil_dofs[0] - self.aligned_eps:
                """
                Handle special case for symmetric BC on the left side
                """
                assert self._src_grid.boundaries[0].type == "symmetric", "This case should only exist for symmetric cases"
                
                """
                src_range starts "too late" which we will fix here
                """
                src_range -= 1

        
        
        """
        Use special 'src_x_dofs' array to store x coordinates for computing FD weights.
        This is required due to the periodic and symmetric boundary conditions.
        """
        src_x_dofs = None
        src_idx_shifted = False
        
        
        """
        Flag to signal whether boundary condition was already handled
        """
        bc_handled = False


        """
        Stencil multipliers
        
        This is useful for symmetric condition if flipping the sign for symmetric effects
        """
        stencil_factors = np.ones(len(src_range))
        
        """
        Given
            src_range
        
        which stores the desired array index range, generate
            src_x_dofs: x-coordinates of DoFs
        
        and 
            src_range: Corresponding index
        """
        if self._src_grid.boundaries[0].type == "periodic":
            
            src_range_wrapped = src_range % self._src_grid.num_dofs
            src_x_dofs = self._src_grid.x_dofs[src_range_wrapped]
            
            range_dist = self._src_grid.domain_end - self._src_grid.domain_start
            
            # Fix wrong x coordinates due to wrapping
            # e.g. -0.2 will be changed to -0.2+range_dist
            src_x_dofs -= (src_range < 0).astype(int)*range_dist
            src_x_dofs += (src_range >= self._src_grid.num_dofs).astype(int)*range_dist
            
            src_range = src_range_wrapped
            assert np.all((src_x_dofs[1:] - src_x_dofs[:-1]) > 0)
            
            bc_handled = True
        
        
        if not bc_handled and self._src_grid.boundaries[0].type == "symmetric":
                        
            if src_range[0] < 0:
                
                """
                Generate flag fields for parts which can be directly used and non-symmetric parts
                """
                flat_nosym = (src_range < 0).astype(int)
                flat_direct = (src_range >= 0).astype(int)
                
                if self._src_grid.boundaries[0].flip_sign:
                    stencil_factors = -1*flat_nosym + flat_direct
                    stencil_factors = stencil_factors.astype(float)
                
                if not self._src_grid.staggered:
                    """
                    Grid point is aligned with the domain start
                    => Do not duplicate point on domain boundary 
                    """
                    
                    """
                    DoF indices
                    """
                    axis_idx = 0
                    src_range_symmetric = (2*axis_idx - src_range)*flat_nosym + src_range*flat_direct
                    assert np.all(src_range_symmetric >= 0)
                    
                else:
                    """
                    Grid point is aligned with the domain start
                    => Duplicate point on domain boundary 
                    
                    Before we start, we figure out whether the first DoF of the first grid is located
                    after the evaluation point and shift the indices for an appropriate handling
                    """
                    
                    """
                    DoF indices
                    """
                    axis_idx = 0
                    src_range_symmetric = (2*axis_idx - src_range - 1)*flat_nosym + src_range*flat_direct
                    assert np.all(src_range_symmetric >= 0)
                
                """
                DoF coordinates
                """
                axis_x = self._src_grid.domain_start
                src_x_dofs = (2.0*axis_x - self._src_grid.x_stencil_dofs[src_range_symmetric])*flat_nosym + self._src_grid.x_stencil_dofs[src_range_symmetric]*flat_direct
                
                if 0:
                    print("x_stencil_dofs: "+str(self._src_grid.x_stencil_dofs))
                    print("+"*80)
                    print("src_range: "+str(src_range))
                    print("src_range_symmetric: "+str(src_range_symmetric))
                    print("src_x_dofs: "+str(src_x_dofs))
                    print("+"*80)
                    #raise Exception("TODO")
                
                src_range = src_range_symmetric
                
                bc_handled = True

        
        if not bc_handled and self._src_grid.boundaries[1].type == "symmetric":
            
            if src_range[-1] >= self._src_grid.num_stencil_grid_points:
                
                """
                Generate flag fields for parts which can be directly used and non-symmetric parts
                """
                flat_direct = (src_range < self._src_grid.num_stencil_grid_points).astype(int)
                flat_nosym = (src_range >= self._src_grid.num_stencil_grid_points).astype(int)
                
                if self._src_grid.boundaries[1].flip_sign:
                    stencil_factors = -1*flat_nosym + flat_direct
                    stencil_factors = stencil_factors.astype(float)
                
                if not self._src_grid.staggered:
                    """
                    DoF indices
                    """
                    axis_idx = self._src_grid.num_stencil_grid_points - 1
                    src_range_symmetric = (2*axis_idx - src_range)*flat_nosym + src_range*flat_direct
                    assert np.all(src_range_symmetric >= 0)
                
                else:
                    """
                    Grid point is aligned with the domain start
                    => Duplicate point on domain boundary 
                    
                    Before we start, we figure out whether the first DoF of the first grid is located
                    after the evaluation point and shift the indices for an appropriate handling
                    """
                    
                    """
                    DoF indices
                    """
                    axis_idx = self._src_grid.num_stencil_grid_points - 1
                    src_range_symmetric = (2*axis_idx - src_range + 1)*flat_nosym + src_range*flat_direct
                    assert np.all(src_range_symmetric >= 0)
                
                """
                DoF coordinates
                """
                axis_x = self._src_grid.domain_end
                #src_x_dofs = (2.0*axis_x - self._src_grid.x_dofs[src_range_symmetric])*flat_nosym + self._src_grid.x_dofs[src_range_symmetric]*flat_direct
                src_x_dofs = (2.0*axis_x - self._src_grid.x_stencil_dofs[src_range_symmetric])*flat_nosym + self._src_grid.x_stencil_dofs[src_range_symmetric]*flat_direct


                if 0:
                    print("+"*80)
                    print("self._src_grid.num_dofs: "+str(self._src_grid.num_dofs))
                    print("src_range: "+str(src_range))
                    print("src_range_symmetric: "+str(src_range_symmetric))
                    print("src_x_dofs: "+str(src_x_dofs))
                    print("+"*80)
                    #raise Exception("TODO")
                
                src_range = src_range_symmetric
                
                bc_handled = True
        


        if self._src_grid.boundaries[0].type == "dirichlet" or self._src_grid.boundaries[0].type == "neumann_extrapolated":
            assert self._src_grid.x_stencil_dofs[1] == self._src_grid.x_dofs[0], "Internal Error"
        
        if not bc_handled:
            """
            Handle general cases of Boundary conditions
            """
            
            # We will work with x_stencil_dofs, hence we first shift everything by 1
            # src_range = [i+1 for i in src_range]

            # Shift src range to right to stay within domain
            if src_range[0] < 0:
                s = -src_range[0]
                src_range = [i+s for i in src_range]
                src_idx_shifted = True
        
            # Shift src range to left to stay within domain
            if src_range[-1] >= len(self._src_grid.x_stencil_dofs):
                s = src_range[-1] - len(self._src_grid.x_stencil_dofs) + 1
                src_range = [i-s for i in src_range]
                src_idx_shifted = True
            
            bc_handled = True
            
            src_x_dofs = self._src_grid.x_stencil_dofs[src_range]
            
            assert src_range[0] >= 0, "Negative index in src_range"
            assert src_range[-1] < self._src_grid.num_dofs+2, "Index too large in src_range"
        
        
        if not bc_handled:
            raise Exception("INTERNAL ERROR: Boundary condition not handled!")

        if 0:
            print("*"*80)
            print("diff_order: "+str(diff_order))
            print("num_stencil_default_offset: "+str(num_stencil_default_offset))
            print("dst_grid.num_dofs: "+str(self._dst_grid.num_dofs))
            print("dst_x: "+str(dst_x))
            print("self._src_grid.num_dofs: "+str(self._src_grid.num_dofs))
            print("src_idx_closest: "+str(src_idx_closest))
            print("src_range: "+str(src_range))
            print("src_x_dofs: "+str(src_x_dofs))
            print("")
            #sys.exit(1)
        
        """
        Validation: Next we check that things are really aligned, e.g., we don't make an extrapolation accidentally
        """
        if not src_idx_shifted:
            
            def internal_error():
                print("src_x_dofs: "+str(src_x_dofs))
                print("dst_x: "+str(dst_x))
                raise Exception("Internal error")
        
            if aligned_grid_point:
                if np.abs(src_x_dofs[len(src_x_dofs)//2]-dst_x) > self.aligned_eps*self.domain_max_abs_val:
                    internal_error()

            else:
                if src_x_dofs[len(src_x_dofs)//2-1] >= dst_x:
                    internal_error()

                if src_x_dofs[len(src_x_dofs)//2] <= dst_x:
                    internal_error()
        
        if dst_x_domain_periodic_shifted:
            src_x_dofs -= self._src_grid.domain_size
            if 0:
                print("src_x_dofs: "+str(src_x_dofs))
        
        
        assert not np.any((src_x_dofs[1:] - src_x_dofs[:-1]) < 0), "Internal error"
        
        return src_range, src_x_dofs, src_idx_closest, stencil_factors

    
    def setup(
        self,
        diff_order,
        dst_grid,
        src_grid = None,
        min_approx_order = None,
        name = None,
    ):
        """
        diff_order:
            Order or differential operator
        
        min_approx_order:
            Approximation order of differential operator
        
        dst_grid:
            Destination variable on which the differential evaluations are computed for

        src_grid:
            Source variable on which the differential evaluations are computed on
        """
        
        if min_approx_order == None:
            min_approx_order = _default_params['operator_diff__min_approx_order']
        
        if src_grid is None:
            src_grid = dst_grid
        
        if name is None:
            name = "OperatorDiff1D"+str(diff_order)+"_"+str(min_approx_order)
        self._name = name

        self._src_grid = src_grid
        self._dst_grid = dst_grid
        
        """
        Threshold to assume two grid points to match each other
        """
        #domain_size = src_grid.domain_end - src_grid.domain_start
        self.domain_max_abs_val = np.max(np.abs([src_grid.domain_start, src_grid.domain_end]))
        self.aligned_eps = 1e-13*self.domain_max_abs_val
        
        """
        Real minimum approximation order.
        
        The generated stencil might be of higher order than requested.
        
        Please note, that the acutal order can be even higher and that this is just
        the lower threshold.
        """
        self.real_approx_order_equispaced = None
        
        """
        Are all grid points on the destination field perfectly aligned with ones on the source field?
        
        This information is used for convergence tests:
        With diff_order=0 and with all DoF coordinates in the destination field, the result must match exactly.

        IMPORTANT: Grid points of the Dirichlet boundary conditions are included in the DoF grid.
        """
        self.all_dst_dof_points_aligned = True
        
        
        """
        Closest index in source variable
        """
        src_idx_closest = 0
        
        """
        Validation check for periodic boundary conditions
        """
        if src_grid.boundaries[0].type == "periodic" or src_grid.boundaries[1].type == "periodic":
            assert src_grid.boundaries[0].type == src_grid.boundaries[1].type

        if src_grid.boundaries[0].type == "periodic" or src_grid.boundaries[1].type == "periodic" or dst_grid.boundaries[0].type == "periodic" or dst_grid.boundaries[1].type == "periodic":
            assert src_grid.boundaries[0].type == src_grid.boundaries[1].type
            assert src_grid.boundaries[0].type == dst_grid.boundaries[0].type, "If one grid has periodic boundary conditions, then both must have periodic boundary conditions"
        
        if 0:
            print("*"*80)
            print("src_grid.x_stencil_dofs: "+str(self._src_grid.x_stencil_dofs))
            print("dst_grid.x_stencil_dofs: "+str(self._dst_grid.x_stencil_dofs))
            print("src_grid: "+str(self._src_grid.x_dofs))
            print("dst_grid: "+str(self._dst_grid.x_dofs))
            print("*"*80)
        
        
        """
        Allocate temporary storage.
        Note, that all src grid points are included here and that boundary conditions are coped with later on.
        This makes it more flexible.
        """
        self._L_sparse_setup = matrix_setup.matrix_sparse((dst_grid.num_dofs, src_grid.num_stencil_grid_points))
        
        """
        Iterate over all points on destination field.
        We only set up the matrix for these points.
        """
        for dst_idx in range(dst_grid.num_dofs):
            
            """
            Point for which we want to compute the differential operator for
            """
            dst_x = dst_grid.x_dofs[dst_idx]

            src_range, src_x_dofs, src_idx_closest, stencil_factors = self._get_src_info(dst_x, diff_order, min_approx_order, src_idx_closest=src_idx_closest)
            
            """
            Determine stencil values
            """
            if 0:
                print("STENCIL")
                print(" + diff_order: "+str(diff_order))
                print(" + dst_x: "+str(dst_x))
                print(" + src_x_dofs: "+str(src_x_dofs))
                print(" + src_range: "+str(src_range))
            
            stencil = fdwe.get_fd_stencil(
                diff_order,
                x_eval = dst_x,
                x_points = src_x_dofs,
            )
            
            stencil *= stencil_factors
            
            """
            Do this step-by-step, since having duplicated entries in src_range (e.g. in case of symmetric BCs), this doens't work.
            """
            for i in range(len(src_range)):
                if 0:
                    # Hack to avoid issue with += updates on sparse matrices
                    tmp = self._L_sparse_setup[dst_idx, src_range[i]].to_numpy_array()
                    tmp += stencil[i]
                    self._L_sparse_setup[dst_idx, src_range[i]] = tmp
                else:
                    self._L_sparse_setup[dst_idx, src_range[i]] += stencil[i]
            
            
            if 0:
                print("*"*80)
                print("src_grid.x_stencil_dofs: "+str(src_grid.x_stencil_dofs))
                print("dst_grid.x_stencil_dofs: "+str(dst_grid.x_stencil_dofs))
                print("src_grid.x_dofs: "+str(src_grid.x_dofs))
                print("dst_grid.x_dofs: "+str(dst_grid.x_dofs))
                print("diff_order: "+str(diff_order))
                print("dst_x: "+str(dst_x))
                print("src_x_dofs: "+str(src_x_dofs))
                print("src_range: "+str(src_range))
                print("stencil: "+str(stencil))
                print("")
        
        
        if dst_x == src_grid.domain_start:
            if src_grid.boundaries[0].type == "symmetric":
                if not src_grid.boundaries[0].flip_sign:
                    if diff_order == 1:
                        assert np.isclose(0, np.sum(self._L_sparse_setup[0,:].to_numpy_array()))
                        self._L_sparse_setup[0,:] = 0
        
        if dst_x == src_grid.domain_end:
            if src_grid.boundaries[1].type == "symmetric":
                if not src_grid.boundaries[1].flip_sign:
                    if diff_order == 1:
                        assert np.isclose(0, np.sum(self._L_sparse_setup[-1,:].to_numpy_array()))
                        self._L_sparse_setup[-1,:] = 0
        
        
        """
        Boundary conditions
        
        So far, we entirely ignored boundary conditions.
        They are put in place right now.
        """
        
        """
        For the numerical differentiation we might also need an additional vector
        The evaluation then computes
            u = A*x + c
        and we allocate 'c' next.
        """
        self._c = libpdefd_vector_array.vector_array_zeros(dst_grid.num_dofs)
        
        if 0:
            np.set_printoptions(linewidth=180)
            print(self._L_sparse_setup.toarray())
            print(self._c)
            print(src_grid.boundaries[0].dirichlet_value)
            print("*"*80)

        """
        Postprocessing of boundaries
        """
        if src_grid.boundaries[0].type == "periodic":

            """
            We only have to get rid of the last column
            """
            self._L_sparse_setup = self._L_sparse_setup[:,:src_grid.num_dofs]
            

        else:
            """
            First, handle boundary conditions at the beginning of the domain
            """
            if src_grid.boundaries[0].type == "dirichlet":
                col = self._L_sparse_setup.getcol_asarray(0)
                self._c += col*src_grid.boundaries[0].dirichlet_value
                self._L_sparse_setup = self._L_sparse_setup[:,1:]
                assert isinstance(self._L_sparse_setup, matrix_setup.matrix_sparse)
            
            elif src_grid.boundaries[0].type == "neumann_extrapolated":
                
                neumann_diff_order = src_grid.boundaries[0].neumann_diff_order
                neumann_value = src_grid.boundaries[0].neumann_value
                x0 = src_grid.x_stencil_dofs[0]
                
                # Compute derivative stencil values on src grid
                src_range, src_x_dofs, _, _ = self._get_src_info(
                                dst_x = x0,
                                diff_order = neumann_diff_order,
                                min_approx_order = min_approx_order + diff_order - 1,
                                src_idx_closest = 0,
                                main_dof_execution = False,
                            )
                
                assert src_range[0] == 0, "Internal error"
                
                # Get stencil values
                stencil = fdwe.get_fd_stencil(
                    diff_order = neumann_diff_order,
                    x_eval = x0,
                    x_points = src_x_dofs
                )
                
                assert stencil[0] != 0, "Internal error"
                
                for j in range(dst_grid.num_dofs):
                    d = self._L_sparse_setup[j,0]/stencil[0]
                    self._c[j] += d * neumann_value
                    
                    for i in range(1, len(stencil)):
                        self._L_sparse_setup[j,i] -= d*stencil[i]
                
                self._L_sparse_setup = self._L_sparse_setup[:,1:]
            
            
            elif src_grid.boundaries[0].type == "symmetric":
                # Nothing to do
                pass
            
            
            else:
                raise Exception("Boundary condition '"+src_grid.boundaries[0].type+"' not supported")
            
            
            
            """
            Second, handle boundary conditions at the end of the domain
            """
            if src_grid.boundaries[1].type == "dirichlet":
                row = self._L_sparse_setup.getcol_asarray(-1)
                self._c += row*src_grid.boundaries[1].dirichlet_value
                
                self._L_sparse_setup = self._L_sparse_setup[:,:-1]
                assert isinstance(self._L_sparse_setup, matrix_setup.matrix_sparse)
            
            elif src_grid.boundaries[1].type == "neumann_extrapolated":

                neumann_diff_order = src_grid.boundaries[1].neumann_diff_order
                neumann_value = src_grid.boundaries[1].neumann_value
                x0 = src_grid.x_stencil_dofs[-1]

                # Compute derivative stencil values on src grid
                src_range, src_x_dofs, _, _ = self._get_src_info(
                                dst_x = x0,
                                diff_order = neumann_diff_order,
                                min_approx_order = min_approx_order + diff_order - 1,
                                src_idx_closest = 0,
                                main_dof_execution = False,
                            )
                
                assert src_range[-1] == src_grid.num_stencil_grid_points-1, "Internal error"
                
                # Get stencil values
                stencil = fdwe.get_fd_stencil(
                    diff_order = neumann_diff_order,
                    x_eval = x0,
                    x_points = src_x_dofs
                )
                
                assert stencil[-1] != 0, "Internal error"
                
                for j in range(dst_grid.num_dofs):
                    d = self._L_sparse_setup[j,-1]/stencil[-1]
                    self._c[j] += d * neumann_value
                    
                    for i in range(0, len(stencil)-1):
                        self._L_sparse_setup[j,i-len(stencil)] -= d*stencil[i]
                
                self._L_sparse_setup = self._L_sparse_setup[:,:-1]
            
            
            elif src_grid.boundaries[1].type == "symmetric":
                # Nothing to do
                pass
            
            
            else:
                raise Exception("Boundary condition '"+src_grid.boundaries[1].type+"' not supported")
        
        
        """
        Set 'all_dst_dof_points_aligned' to False if not all output DoFs
        are aligned with the source ones 
        """      
        if self.all_dst_dof_points_aligned:
            for i in range(2):
                if src_grid.boundaries[i].type == "periodic":
                    pass
                
                elif src_grid.boundaries[i].type in ['neumann_extrapolated']:
                    if dst_grid.boundaries[i].type in ['symmetric', 'periodic']:
                        if not self._src_grid.staggered:
                            self.all_dst_dof_points_aligned = False
                    
                elif src_grid.boundaries[i].type in ['dirichlet']:
                    pass
                
                elif src_grid.boundaries[i].type in ['symmetric']:
                    pass
            
                else:
                    raise Exception("Case not handled")
        
        
        if 0:
            np.set_printoptions(linewidth=180)
            print(self._L_sparse_setup.to_numpy_array())
            print(self._L_sparse_setup.to_numpy_array())
            print(self._c)

        if 0:
            # Estimate possible cancellation errors
            self._L_sparse_setup_min = self._L_sparse_setup.reduce_min()
            self._L_sparse_setup_max = self._L_sparse_setup.reduce_max()
            self._L_sparse_setup_cancellation_error = self._L_sparse_setup_max - self._L_sparse_setup_min
        
        if 0:
            np.set_printoptions(linewidth=180)
            print(self._L_sparse_setup.to_numpy_array().shape)
            print(self._L_sparse_setup.to_numpy_array())
            print(self._c)
        
        assert self._L_sparse_setup.shape[0] == self._c.shape[0], "Internal Error"
        assert self._L_sparse_setup.shape[0] == dst_grid.num_dofs, "Internal Error"
        


    def __str__(self):
        retstr = ""
        retstr += "Operator matrix:\n"
        retstr += str(self.get_L())
        return retstr



class OperatorDiffND(_operator_base):
    def __init__(self, *args, **kwargs):
        _operator_base.__init__(self)

        self.setup(*args, **kwargs)
    
    def setup(
        self,
        diff_dim: int,
        diff_order : int,
        dst_grid : GridInfoND,
        src_grid  : GridInfoND = None,
        min_approx_order : int = None,
        assert_aligned = False,
    ):
        """
        diff_dim:   Dimension along which to compute the differentiation
        diff_order: Order of differentiation to compute
        min_approx_order:    Minimal order of approximation
        dst_grid:   Target grid
        src_grid:   Source grid
        assert_aligned: Check with assertions that the grids are aligned
                        in all dimension in which no differentiation is computed.
                        Set this to 'False' only if you know what you're doing since
                        you can easily mess things up.
                        
        We do a step-by-step operator application over all dimensions.
        """
        if src_grid == None:
            src_grid = dst_grid
        
        if min_approx_order == None:
            min_approx_order = _default_params['operator_diff__min_approx_order']
        
        self._dim = diff_dim
        self._diff_order = diff_order
        self.min_approx_order = min_approx_order
        self._dst_grid = dst_grid
        self._src_grid = src_grid
        
        assert len(dst_grid) == len(src_grid)
        assert isinstance(dst_grid, GridInfoND)
        assert isinstance(src_grid, GridInfoND)
        
        num_dims = len(dst_grid)
        
        def get_bcast_matrix_L(L, i_dim):
            """
            Generate matrix with the linear operator 'L' in it
            """
            retm = matrix_setup.matrix_sparse(np.array([1]))
            
            for i in range(0, i_dim):
                M = matrix_setup.eye_sparse(dst_grid.shape[i])
                retm = retm.kron(M)
            
            retm = retm.kron(L)
            
            for i in range(i_dim+1, num_dims):
                M = matrix_setup.eye_sparse(src_grid.shape[i])
                #retm = sparse.kron(retm, M)
                retm = retm.kron(M)
            
            return retm
        
        
        def get_bcast_matrix_c(c, i_dim):
            """
            Generate matrix with which we can bcast the 'c' boundary conditions
            """
            retm = libpdefd_vector_array.vector_array([1], dtype='float')
            
            for i in range(0, i_dim):
                M = libpdefd_vector_array.vector_array_ones(dst_grid.shape[i])
                retm = retm.kron_vector(M)
                
            retm = retm.kron_vector(c)
            
            for i in range(i_dim+1, num_dims):
                M = libpdefd_vector_array.vector_array_ones(src_grid.shape[i])
                
                retm = retm.kron_vector(M)

            return retm
        
        
        total_src_N = np.prod(src_grid.shape)
        total_dst_N = np.prod(dst_grid.shape)
        
        
        self._L_sparse_setup = matrix_setup.eye_sparse(total_src_N)
        self._c = libpdefd_vector_array.vector_array_zeros(total_src_N)
        
        if 0:
            print("self._L_sparse_setup: "+str(self._L_sparse_setup.to_numpy_array().shape))
            print("self._c: "+str(self._c.shape))
            print("src_grid.shape:", src_grid.shape)
            print("dst_grid.shape:", dst_grid.shape)
        
        for dim in range(len(dst_grid)):
            if 0:
                print("*"*80)
                print("DIM "+str(dim))
            
            """
            Special differential operator in this particular dimension.
            If the derivative shouldn't be computed along a particular dimension,
            we only use an interpolation
            """
            
            diff_op = OperatorDiff1D(
                    diff_order = self._diff_order if dim == diff_dim else 0,        # Differentiation or interpolation?
                    min_approx_order = min_approx_order,
                    dst_grid = dst_grid[dim],
                    src_grid = src_grid[dim]
                )
        
            if assert_aligned:
                if dim != diff_dim:
                    if np.any(dst_grid.shape[dim] != src_grid.shape[dim]):
                        print("Nonmatching grid info:")
                        print(dst_grid)
                        print(src_grid)
                        raise Exception("Grid in dimension "+str(dim)+" doesn't match, please check")
                    
                    # abs is natively supported, but max isn't
                    err = abs(diff_op._L_sparse_setup - matrix_setup.eye_sparse(np.product(dst_grid[dim].num_dofs))).reduce_max()
                    if err > 1e-10*(abs(diff_op._L_sparse_setup).reduce_max()):
                        print("*"*80)
                        print("* ERROR")
                        print("*"*80)
                        print("Grids in dimension "+str(dim)+" doesn't seem to match")
                        print(" + please check whether you forgot an interpolation")
                        print(" + interpolation is only allwed in a single dimension for setting up an operator")
                        print(" + combine multiple operators in order to concatenate them to a single one")
                        raise Exception("Grids in dimension "+str(dim)+" doesn't seem to match")
            
            """
            Setup operator matrix and 'c' vector which assumes all dimensions up to 'dim' have been already processed.
            """
            
            BL = get_bcast_matrix_L(diff_op._L_sparse_setup, dim)
            
            Bc = get_bcast_matrix_c(diff_op._c, dim)
            
            
            if 0:
                print("diff_op._L_sparse_setup", diff_op._L_sparse_setup.to_numpy_array().shape)
                print("diff_op._c", diff_op._c.shape)
                print("BL", BL.to_numpy_array().shape)
                print("Bc", Bc.shape)
                print("self._c", self._c.shape)
            
            if dim == 0:
                self._L_sparse_setup = BL
                self._c = Bc
            else:
                self._L_sparse_setup = BL.dot(self._L_sparse_setup)
                self._c = BL.dot(self._c) + Bc
            
            
            if self._L_sparse_setup.shape[0] != self._c.shape[0]:
                print("Internal Error")
                print(" + dim: "+str(dim))
                print(" + self._L_sparse_setup.shape: "+str(self._L_sparse_setup.shape))
                print(" + self._c.shape: "+str(self._c.shape))
                raise Exception("Internal Error")
            
            if 0:
                print("INFO")
                print(" + dim: "+str(dim))
                print(" + self._L_sparse_setup.shape: "+str(self._L_sparse_setup.shape))
                print(" + self._c.shape: "+str(self._c.shape))
                print(" + dst_grid.shape: "+str(dst_grid.shape))
                
            ndofs = np.prod(dst_grid.shape)
            assert self._L_sparse_setup.shape[0] == self._c.shape[0], "Internal Error"
        
        assert self._L_sparse_setup.shape[0] == self._c.shape[0], "Internal Error"
        assert self._L_sparse_setup.shape[0] == np.prod(dst_grid.shape), "Internal Error"

