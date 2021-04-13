import numpy as np
import sys

import libpdefd.libpdefd_matrix as libpdefd_matrix
import libpdefd.fd_weights_explicit as fdwe



class _VariableBase:
    """
    Provide basic arithmetic operations for variable
    """
    def __init__(self):
        self.name = "TMP"
        self.data = None
        self.shape = None
    
    def reciprocal(self):
        #name = "("+self.name+"+"+a.name+")"
        name = "TMP"
        data = 1.0/self.data
        return self.__class__(self.grid, name=name, data=data)

    def copy(self):
        return self.__class__(self.grid, name=self.name, data=np.copy(self.data))
        
        
    def __add__(self, a):
        #name = "("+self.name+"+"+a.name+")"
        name = "TMP"
        if isinstance(a, self.__class__):
            data = self.data + a.data
        else:
            data = self.data + a
        return self.__class__(self.grid, name=name, data=data)

    def __sub__(self, a):
        #name = "("+self.name+"+"+a.name+")"
        name = "TMP"
        if isinstance(a, self.__class__):
            data = self.data - a.data
        else:
            data = self.data - a
        return self.__class__(self.grid, name=name, data=data)

    def __mul__(self, a):
        if isinstance(a, self.__class__):
            #name = "("+self.name+"*"+a.name+")"
            name = "TMP"
            data = self.data * a.data
 
        else:
            #name = "("+self.name+"*"+str(a)+")"
            name = "TMP"
            data = self.data * a
            
        return self.__class__(self.grid, name=name, data=data)


    def __rmul__(self, a):
        """
        Support
            scalar * VariableND()
        operators
        """
        if isinstance(a, self.__class__):
            #name = "("+self.name+"*"+a.name+")"
            name = "TMP"
            data = a.data * self.data
 
        else:
            #name = "("+self.name+"*"+str(a)+")"
            name = "TMP"
            data = a * self.data
            
        return self.__class__(self.grid, name=name, data=data)


    def __truediv__(self, a):
        """
        Support regular division (not floored/integer division)
        """
        #name = "("+self.name+"/"+a.name+")"
        name = "TMP"
        if isinstance(a, self.__class__):
            data = self.data / a.data
        else:
            data = self.data / a
        return self.__class__(self.grid, name=name, data=data)


    def __rtruediv__(self, a):
        """
        Support regular division (not floored/integer division)
        """
        #name = "("+self.name+"/"+a.name+")"
        name = "TMP"
        if isinstance(a, self.__class__):
            data = a.data / self.data
        else:
            data = a / self.data
        return self.__class__(self.grid, name=name, data=data)

    def __iadd__(self, a):
        """
        Support
            +=
        operators
        """
        if isinstance(a, self.__class__):
            self.data += a.data
        else:
            self.data += a
        return self

    def __isub__(self, a):
        if isinstance(a, self.__class__):
            self.data -= a.data
        else:
            self.data -= a
        return self

    def __imul__(self, a):
        if isinstance(a, self.__class__):
            self.data *= a.data
        else:
            self.data *= a
        return self

    def __idiv__(self, a):
        if isinstance(a, self.__class__):
            self.data /= a.data
        else:
            self.data /= a
        return self

    def __pow__(self, a):
        if isinstance(a, self.__class__):
            #name = "("+self.name+"*"+a.name+")"
            name = "TMP"
            data = self.data ** a.data
 
        else:
            #name = "("+self.name+"*"+str(a)+")"
            name = "TMP"
            data = self.data ** a
            
        return self.__class__(self.grid, name=name, data=data)

    def __neg__(self):
        #name = "(-"+self.name+")"
        name = "TMP"
        data = -self.data
        return self.__class__(self.grid, name=name, data=data)

    def __pos__(self):
        #name = "("+self.name+")"
        name = "TMP"
        data = self.data
        return self.__class__(self.grid, name=name, data=data)


    def set_all(
        self,
        value
    ):
        self.data[:] = value



class Variable1D(_VariableBase):
    def __init__(self, *args, **kwargs):
        _VariableBase.__init__(self)
        
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        grid,
        name = None,
        data = None
    ):
        self.name = name
        if self.name is None:
            self.name = "VAR"
        
        self.grid = grid
       
        self.data = data
        if self.data is None:
            self.data = np.zeros_like(self.grid.x_dofs)

        self.shape = self.grid.x_dofs.shape

    def __getitem__(self, i):
        return self.data[i]


    def __setitem__(self, i, data):
        self.data[i] = data


    def set(
        self,
        a
    ):
        if isinstance(a, Variable1D):
            assert self.grid.x_dofs.shape == a.data.shape, "Shape mismatch"
            self.data = a.data
        else:
            assert self.grid.x_dofs.shape == a.shape, "Shape mismatch"
            self.data = a


    def __str__(self):
        retstr = ""
        retstr += self.name
        return retstr



class VariableND(_VariableBase):
    def __init__(self, *args, **kwargs):
        _VariableBase.__init__(self)
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        grid,
        name = None,
        data = None
    ):
        assert isinstance(grid, GridInfoND), "Only GridInfoND supported for grid parameter"
        
        self.name = name
        if self.name is None:
            self.name = "VAR"
        
        self.grid = grid
        self.data = data
        self.shape = self.grid.shape

        if self.data is None:
            self.data = np.zeros(self.shape)
    
    
    def __getitem__(self, i):
        return self.data[i]


    def __setitem__(self, i, data):
        self.data[i] = data


    def set(
        self,
        a
    ):
        if isinstance(a, Variable1D):
            assert self.grid.x_dofs.shape == a.data.shape, "Shape mismatch"
            self.data = a.data
        
        elif isinstance(a, np.ndarray):
            assert self.shape == a.shape, "Shape mismatch: "+str(self.shape)+" != "+str(a.shape)
            self.data = a
            
        else:
            raise Exception("Type '"+str(type(a))+"' not supported")


            
    def __str__(self):
        retstr = ""
        retstr += "variable: "+str(self.name)
        retstr += ", shape: "+str(self.shape)
        return retstr



def VariableNDSet_Empty_Like(variable_set_like: list):
    return VariableNDSet([None for _ in range(len(variable_set_like.variable_list))])



class VariableNDSet:
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    
    def setup(
        self,
        variable_list: list,
    ):
        if not isinstance(variable_list, list):
            if not isinstance(variable_list, Variable1D) and not isinstance(variable_list, VariableND):
                raise Exception("Only Variables are supported")
            
            variable_list = [variable_list]

        self.variable_list = variable_list
    
    
    def copy(self):
        v = VariableNDSet([i.copy() for i in self.variable_list])
        return v
    
    
    def __getitem__(self, key):
        
        if isinstance(key, str):
            for i in range(len(self.variable_list)):
                if self.variable_list[i].name == key:
                    return self.variable_list[i]
                
            raise Exception("Field '"+str(key)+"' not found in set of variables")
        
        return self.variable_list[key]
    
    
    def __setitem__(self, key, data):
        
        if isinstance(key, str):
            for i in range(len(self.variable_list)):
                if self.variable_list[i].name == key:
                    self.variable_list[i] = data
                    return
            
            raise Exception("Field '"+str(key)+"' not found in set of variables")
        
        self.variable_list[key] = data


    def set(
        self,
        variable_list_or_class
    ):
        if isinstance(variable_list_or_class, list):
            self.variable_list[:] = variable_list_or_class[:]
        
        elif isinstance(variable_list_or_class, VariableNDSet):
            self.variable_list[:] = variable_list_or_class.variable_list[:]
                
        else:
            raise Exception("TODO")


    def __add__(self, a):
        return VariableNDSet([self.variable_list[i] + a.variable_list[i] for i in range(len(a.variable_list))])

    def __sub__(self, a):
        return VariableNDSet([self.variable_list[i] - a.variable_list[i] for i in range(len(a.variable_list))])

    def __mul__(self, a):
        if isinstance(a, Variable1D):
            return VariableNDSet([self.variable_list[i] * a.variable_list[i] for i in range(len(self.variable_list))])
 
        else:
            return VariableNDSet([self.variable_list[i] * a for i in range(len(self.variable_list))])


    def __rmul__(self, a):
        if isinstance(a, Variable1D):
            return VariableNDSet([a.variable_list[i] * self.variable_list[i] for i in range(len(self.variable_list))])
 
        else:
            return VariableNDSet([a * self.variable_list[i] for i in range(len(self.variable_list))])


    def __div__(self, a):
        return VariableNDSet([self.variable_list[i] / a.variable_list[i] for i in range(len(a.variable_list))])

    def __iadd__(self, a):
        for i in range(len(self.variable_list)):
            self.variable_list[i] += a.variable_list[i]
        return self

    def __isub__(self, a):
        for i in range(len(self.variable_list)):
            self.variable_list[i] -= a.variable_list[i]
        return self

    def __imul__(self, a):
        for i in range(len(self.variable_list)):
            self.variable_list[i] *= a.variable_list[i]
        return self

    def __idiv__(self, a):
        for i in range(len(self.variable_list)):
            self.variable_list[i] /= a.variable_list[i]
        return self

    def __neg__(self):
        return VariableNDSet([-v for v in self.variable_list])

    def __pos__(self):
        return VariableNDSet([v for v in self.variable_list])

    def __str__(self):
        retstr = "VariableNDSet:\n"
        for i in range(len(self.variable_list)):
            retstr += " + "+str(self.variable_list[i].name)+": shape="+str(self.variable_list[i].data.shape)+"\n"
        
        return retstr




class Boundary:
    def __init__(self, type):
        self.type = type
        
    def __str__(self):
        retstr = ""
        retstr += self.type
        return retstr


class BoundaryPeriodic(Boundary):
    def __init__(self):
        Boundary.__init__(self, "periodic")

    def __str__(self):
        retstr = ""
        retstr += self.type
        return retstr


class BoundaryDirichlet(Boundary):
    def __init__(self, dirichlet_value):
        Boundary.__init__(self, "dirichlet")
        self.dirichlet_value = dirichlet_value

    def __str__(self):
        retstr = ""
        retstr += self.type+"("+str(self.dirichlet_value)+")"
        return retstr


class BoundaryNeumannExtrapolated(Boundary):
    """
    Neumann boundary without DoF on Boundary
    
    This BC relates to reconstruct a Neumann DoF of 0th order
    which matches the required derivative.
    
    The difference to the previous boundary condition is that
    there *exists no DoF at the boundary*. 
    """
    def __init__(self, neumann_value = 0, diff_order = 1):
        Boundary.__init__(self, "neumann_extrapolated")
        self.neumann_value = neumann_value
        
        if diff_order <= 0:
            raise Exception("Neumann derivative only for >0 order")
        
        self.neumann_diff_order = diff_order

    def __str__(self):
        retstr = ""
        retstr += self.type+"("+str(self.neumann_diff_order)+")"
        return retstr



class BoundarySymmetric(Boundary):
    """
    Make the boundary symmetric which means that we have Neumann df/dx*n=0 boundary conditions
    
    This BC relates to a DoF on the boundaries at which the
    given order and value is fulfilled.
    
    The difference to the previous boundary condition is that
    there exists a DoF at this boundary. 
    """
    def __init__(self, flip_sign = False):
        Boundary.__init__(self, "symmetric")
        self.flip_sign = flip_sign
        
    def __str__(self):
        retstr = ""
        retstr += self.type
        return retstr



class GridInfo1D:
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(self,
              name = None,
              dim = 0
    ):
        self.name = name
        self.dim = dim
        
        """
        Coordinates of DoFs including those for "Dirichlet", "Neumann" and "Periodic" boundary conditions.
        They will be used for generation of stencil operators.
        """
        self.x_stencil_dofs = None
        
        """
        Coordinates of DoFs.
        This will also be the dimension of the final linear operator.
        """
        self.x_dofs = None
        
        assert self.dim >= 0, "Dimension mismatch"


    def _setup_common(
        self
    ):
        if not isinstance(self.boundaries[0], BoundaryPeriodic):
            assert self.x_stencil_dofs[0] == self.domain_start, "First coordinate must coincice with domain start"
            assert self.x_stencil_dofs[-1] == self.domain_end, "Last coordinate must coincice with domain end"
        
        
        """
        Fix DoFs for stencil sources
        
        Only required for symmetric boundary conditions for staggered grids
        """
        if self.boundaries[0].type == "symmetric":
            if self.staggered:
                assert self.x_stencil_dofs[0] == self.domain_start
                self.x_stencil_dofs = self.x_stencil_dofs[1:]
            
        if self.boundaries[1].type == "symmetric":
            if self.staggered:
                assert self.x_stencil_dofs[-1] == self.domain_end
                self.x_stencil_dofs = self.x_stencil_dofs[:-1]
    
        """
        Determine grid for degrees of freedom
        """
        self.x_dofs = np.copy(self.x_stencil_dofs)
        
        """
        Is first coordinate in x_stencil_dofs field a real DoF?
        """
        #self.first_x_bc_is_real_dof = False

        if self.boundaries[0].type == "periodic":
            pass
            
        elif self.boundaries[0].type == "dirichlet":
            self.x_dofs = self.x_dofs[1:]
            
        elif self.boundaries[0].type == "neumann_extrapolated":
            self.x_dofs = self.x_dofs[1:]
            
        elif self.boundaries[0].type == "symmetric":
            pass
            
        else:
            raise Exception("Boundary condition '"+self.boundaries[0].type+"' not supported")
        
        
        if self.boundaries[1].type == "periodic":
            self.x_dofs = self.x_dofs[:-1]
            
        elif self.boundaries[1].type == "dirichlet":
            self.x_dofs = self.x_dofs[:-1]
            
        elif self.boundaries[1].type == "neumann_extrapolated":
            self.x_dofs = self.x_dofs[:-1]
            
        elif self.boundaries[1].type == "symmetric":
            pass
            
        else:
            raise Exception("Boundary condition '"+self.boundaries[1].type+"' not supported")
    
        self.num_dofs = len(self.x_dofs)
        self.num_stencil_grid_points = len(self.x_stencil_dofs)
    
    
    
    def setup_autogrid(
        self,
        x_start,
        x_end,
        regular_num_grid_points,
        x_distribution = "equidistant",
        x_distribution_lambda = None,
        boundaries = [None, None],
        staggered = False
    ):
        """
        x_start:
            start coordinate of grid
            
        x_end:
            end coordinate of grid
            
        regular_num_stencil_grid_points:
            suggested number of degrees of freedom (nodes for finite differences) for regular grid layout.
            Note, that this can change internally depending on grid staggering without periodic boundaries. 
            
        x_distribution:
            'equidistant': same space between nodes
            
        x_distribution_lambda:
            Lambda function to compute coordinate
            [0;1] -> |R
            
        boundaries:
            Boundary condition. First element: left boundary, Second element: right boundary
            
        staggered:
            Staggered grid layout
        """
        
        self.domain_start = x_start
        self.domain_end = x_end
        self.domain_size = self.domain_end - self.domain_start
        
        self.distribution = x_distribution
        self.distribution_lambda = x_distribution_lambda
        self.boundaries = boundaries
        
        if self.boundaries[0] is None:
            self.boundaries[0] = BoundaryPeriodic()
        
        if self.boundaries[1] is None:
            self.boundaries[1] = BoundaryPeriodic()
        
        self.staggered = staggered
        
        """
        Boundary conditions
        """
        assert len(self.boundaries) == 2, "There must be exactly 2 boundaries provided (the one at the beginning and the end of the domain)"

        # Sanitation checks on boundaries
        if self.boundaries[0].type == "periodic":
            assert self.boundaries[1].type == "periodic", "Either both boundaries must be periodic or none"
        
        
        if not staggered:
            x_std = np.linspace(x_start, x_end, regular_num_grid_points, endpoint=True)

        else:
            """
            Staggered grid
            """

            """
            First, generate a half-shifted grid
            """
            x_half_shifted = np.linspace(x_start, x_end, regular_num_grid_points, endpoint=True)
            x_half_shifted += 0.5*(x_half_shifted[1] - x_half_shifted[0])
            
            if self.boundaries[0].type == "periodic":
                x_std = x_half_shifted
            
            else:
                """
                This grid will have an additional DoF!
                """
                regular_num_grid_points += 1
                x_std = np.empty(regular_num_grid_points)
                
                x_std[0] = x_start
                x_std[1:-1] = x_half_shifted[:-1]
                x_std[-1] = x_end
                
                
        
        """
        Generate grid. This includes the grid points on the boundaries.
        """
        if x_distribution == "equidistant":
            self.x_stencil_dofs = x_std
        
        elif x_distribution == "lambda":
            self.x_stencil_dofs = x_distribution_lambda(x_std)

        else:
            raise Exception("Distribution "+str(x_distribution)+" is not supported")

        self._setup_common()
    
    
    def setup_manualgrid(
        self,
        x_stencil_dofs,
        x_start = None,
        x_end = None,
        boundaries = [None, None],
        staggered = False
    ):
        """
        x_stencil_dofs:
            all grid points
            + excluding boundary ones for periodic and symmetric
              boundary conditions if using grid staggering
            + including boundary ones for all other boundary conditions
            
        x_start:
            start coordinate of grid
            
        x_end:
            end coordinate of grid
            
        x_distribution:
            'equidistant': same space between nodes
            
        x_distribution_lambda:
            Lambda function to compute coordinate
            [0;1] -> |R
            
        boundaries:
            Boundary condition. First element: left boundary, Second element: right boundary
            
        name:
            Name to assiciate with this variable
        """
        
        if x_start is None:
            x_start = x_stencil_dofs[0]

        if x_end is None:
            x_end = x_stencil_dofs[-1]

        self.domain_start = x_start
        self.domain_end = x_end

        self.domain_size = self.domain_end - self.domain_start
        self.boundaries = boundaries
        
        self.staggered = staggered
        
        assert self.boundaries[0] != None, "Please specify boundary condition at beginning of domain"
        assert self.boundaries[1] != None, "Please specify boundary condition at end of domain"
        
        
        """
        Boundary conditions
        """
        assert len(self.boundaries) == 2, "There must be exactly 2 boundaries provided (the one at the beginning and the end of the domain)"

        # Sanitation checks on boundaries
        if self.boundaries[0].type == "periodic":
            assert self.boundaries[1].type == "periodic", "Either both boundaries must be periodic or none"
        
        
        """
        Generate grid
        """
        self.x_stencil_dofs = x_stencil_dofs

        self._setup_common()


    def _str_array_pretty(self, x):
        if len(x) <= 16:
            return str(x)

        retval = "["
        retval += ", ".join([str(i) for i in x[0:4]])
        retval += ", ..., "
        retval += ", ".join([str(i) for i in x[-4:]])
        retval += "]"
        return retval

        
    def __str__(self):
        retval = ""
        retval += "GridInfo1D '"+self.name+"'\n"
        retval += " + domain boundaries: ["+str(self.domain_start)+", "+str(self.domain_end)+"]\n"
        retval += " + num_dofs: "+str(self.num_dofs)+ "\n"
        retval += " + x_dofs: "+self._str_array_pretty(self.x_dofs)+"\n"
        retval += " + x_stencil_dofs: "+self._str_array_pretty(self.x_stencil_dofs)+"\n"
        retval += " + boundary_left: "+str(self.boundaries[0])+"\n"
        retval += " + boundary_right: "+str(self.boundaries[1])+"\n"
        return retval
    
    
    def _str_compact(self):
        retval = "domain: ["+str(self.domain_start)+", "+str(self.domain_end)+"]"
        retval += ", num_dofs: "+str(self.num_dofs)
        retval += ", boundaries: "+str(self.boundaries[0])+", "+str(self.boundaries[1])
        return retval



class GridInfoND:
    name_counter = 0
    
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        grids1d_list : list,
        name = None
    ):
        self.grids1d_list = grids1d_list
        self.name = name
        
        if self.name is None:
            self.name = "grid_info_nd_"+str(GridInfoND.name_counter)
            GridInfoND.name_counter += 1
        
        for i in range(len(self.grids1d_list)):
            if self.grids1d_list[i].dim != None:
                assert self.grids1d_list[i].dim == i, "Mismatch between grid and dimension"
        
        self.shape = tuple([grid.num_dofs for grid in self.grids1d_list])
    
    
    def get_meshcoords_nonstaggered(self, sparse=False):
        return [grid.x_dofs for grid in self.grids1d_list]
    
    
    def get_plotting_mesh(self, sparse=False):
        """
        Return tuple of mesh coordinates
        First tuple element is for first dimension, second one for second dimension, etc. 
        """
        return np.meshgrid(*self.get_meshcoords_nonstaggered(), sparse=sparse, indexing="ij")
    
    
    def get_meshcoords_staggered(self, sparse=False):
        """
        Return mesh discrete axis points suitable for
        matplotlib.pyplot.pcolormesh
        """
        mesh_grid_coords = []
        
        for i in range(len(self.grids1d_list)):
            x_dofs = self.grids1d_list[i].x_dofs
            
            mesh_grid_coords.append(np.zeros(len(x_dofs)+1))
            m = mesh_grid_coords[-1]
            
            dx = x_dofs[1] - x_dofs[0]
            m[0] = x_dofs[0] - 0.5*dx
            m[1:-1] = 0.5*(x_dofs[1:] + x_dofs[:-1])
            dx = x_dofs[-1] - x_dofs[-2]
            m[-1] = x_dofs[-1] + 0.5*dx
        
        return mesh_grid_coords


    def get_mesh(self, sparse=False):
        """
        Return tuple of mesh coordinates
        First tuple element is for first dimension, second one for second dimension, etc. 
        """
        return np.moveaxis(np.array(self.get_plotting_mesh(sparse=sparse)), 0, -1)
    
    
    def __getitem__(self, i):
        
        if isinstance(i, str):
            
            for g in self.grids1d_list:
                if g.name == i:
                    return g
                
            raise Exception("GridInfoND '"+str(i)+"' not found")
        
        else:
            return self.grids1d_list[i]

    def __setitem__(self, i, data):
        self.grids1d_list[i] = data

    def __len__(self):
        return len(self.grids1d_list)

    def __str__(self):
        retstr = "GridInfoND: "+self.name+"\n"
        for i in range(len(self.grids1d_list)):
            retstr += " + dim: "+str(i)
            retstr += ", "+self.grids1d_list[i]._str_compact()
            retstr += "\n"
            
        return retstr



class GridInfoNDSet:
    name_counter = 0
    
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        grid_info_nd_list : list,
        name = None
    ):
        self.grid_info_nd_list = grid_info_nd_list
        self.name = name
        
        if self.name is None:
            self.name = "grid_info_nd_set_"+str(GridInfoNDSet.name_counter)
            GridInfoNDSet.name_counter += 1
        
        for i in range(len(self.grid_info_nd_list)):
            assert isinstance(self.grid_info_nd_list[i], GridInfoND)
    
    
    def __getitem__(self, key):
        
        if isinstance(key, str):
            for i in range(len(self.grid_info_nd_list)):
                if self.grid_info_nd_list[i].name == key:
                    return self.grid_info_nd_list[i]
                
            raise Exception("Field '"+str(key)+"' not found in set of grid infos")
        
        return self.grid_info_nd_list[key]
    
    
    def __setitem__(self, key, data):
        
        if isinstance(key, str):
            for i in range(len(self.grid_info_nd_list)):
                if self.grid_info_nd_list[i].name == key:
                    self.grid_info_nd_list[i] = data
                    return
            
            raise Exception("Field '"+str(key)+"' not found in set of grid infos")
        
        self.grid_info_nd_list[key] = data


    def __len__(self):
        return len(self.grid_info_nd_list)

    def __str__(self):
        retstr = "GridInfoNDSet: "+self.name+"\n"
        for i in range(len(self.grid_info_nd_list)):
            retstr += " + "+str(self.grid_info_nd_list[i].name)+"\n"
        
        return retstr



class MeshND:
    """
    Create mesh from grid
    """
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        grid_info_nd: GridInfoND,
        name = None
    ):
        self.grid_info_nd = grid_info_nd
        self.name = name
        
        self.data = grid_info_nd.get_mesh()
        
        if name is None:
            self.name = grid_info_nd.name



class MeshNDSet:
    name_counter = 0

    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        mesh_or_grid_nd_list : list,
        name = None
    ):
        self.name = name
        
        if self.name is None:
            self.name = "mesh_nd_set_"+str(MeshNDSet.name_counter)
            MeshNDSet.name_counter += 1
        

        if isinstance(mesh_or_grid_nd_list[0], GridInfoND):
            """
            Greate MeshNDSet based on GridInfoND
            """
            _grid_info_nd_list = [MeshND(i) for i in mesh_or_grid_nd_list]
            self.mesh_nd_list = MeshNDSet(_grid_info_nd_list)
        
        else:
            """
            MeshNDSet already exists
            """
            self.mesh_nd_list = mesh_or_grid_nd_list
            for i in range(len(self.mesh_nd_list)):
                assert isinstance(self.mesh_nd_list[i], MeshND)
    
    
    def __getitem__(self, key):
        
        if isinstance(key, str):
            for i in range(len(self.mesh_nd_list)):
                if self.mesh_nd_list[i].name == key:
                    return self.mesh_nd_list[i]
                
            raise Exception("Field '"+str(key)+"' not found in set of mesh sets")
        
        return self.mesh_nd_list[key]
    
    
    def __setitem__(self, key, data):
        
        if isinstance(key, str):
            for i in range(len(self.mesh_nd_list)):
                if self.mesh_nd_list[i].name == key:
                    self.mesh_nd_list[i] = data
                    return
            
            raise Exception("Field '"+str(key)+"' not found in set of grid infos")
        
        self.mesh_nd_list[key] = data


    def __len__(self):
        return len(self.grid_info_nd_list)

    def __str__(self):
        retstr = "MeshNDSet: "+self.name+"\n"
        for i in range(len(self.mesh_nd_list)):
            retstr += " + "+str(self.mesh_nd_list[i].name)+": shape="+str(self.mesh_nd_list[i].data.shape)+"\n"
        
        return retstr



class _OperatorBase:
    def __init__(self):
        self.L_sparse = None
        self.name = "TMP"
    
    def setup(self, name):
        self.name = name
    
    def get_L(self):
        return self.L_sparse.toarray()
    
    def get_c(self):
        return self.c
    
    def get_L_sparse(self):
        return self.L_sparse
    
    def solve(self, x, solver_tol=None, max_iterations=None):
        if isinstance(x, np.ndarray):
            raise Exception("TODO")
        
            import libtide.linalg.iterative_solvers as iterative_solvers
            b = x.flatten()-self.c

            retval = iterative_solvers.solve(self.L_sparse, b, solver="gmres", solver_tol=solver_tol, max_iterations=max_iterations)

            return retval.reshape(self.dst_grid.shape)
            #return self.L_sparse.dot(x) + self.c
        
        elif isinstance(x, Variable1D):
            raise Exception("TODO")
            return Variable1D(
                self.dst_grid,
                name = self.name+"("+x.name+")",
                data = self.L_sparse.dot(x.data) + self.c
            )
        
        elif isinstance(x, VariableND) or isinstance(x, _VariableBase):
            raise Exception("TODO")
            assert self.L_sparse.shape[0] == self.c.shape[0]
            
            data = (self.L_sparse.dot(x.data.flatten()) + self.c).reshape(self.dst_grid.shape)
            
            return VariableND(
                self.dst_grid,
                name = self.name+"("+x.name+")",
                data = data
            )
        
        raise Exception("Don't know how to handle '"+str(type(x))+"'")
        
    def apply(self, x):
        if isinstance(x, np.ndarray):
            return self.L_sparse.dot(x) + self.c
        
        elif isinstance(x, Variable1D):
            return Variable1D(
                self.dst_grid,
                name = self.name+"("+x.name+")",
                data = self.L_sparse.dot(x.data) + self.c
            )
        
        elif isinstance(x, VariableND) or isinstance(x, _VariableBase):
            assert self.L_sparse.shape[0] == self.c.shape[0]
            
            data = (self.L_sparse.dot(x.data.flatten()) + self.c).reshape(self.dst_grid.shape)
            
            return VariableND(
                self.dst_grid,
                name = self.name+"("+x.name+")",
                data = data
            )
        
        raise Exception("Don't know how to handle '"+str(type(x))+"'")
    
    def __call__(self, x):
        return self.apply(x)
    
    def __add__(self, a):
        retval = self.__class__()
        retval.L_sparse = self.L_sparse + a.L_sparse
        return retval

    def __sub__(self, a):
        retval = self.__class__()
        retval.L_sparse = self.L_sparse - a.L_sparse
        return retval

    def __mul__(self, a):
        retval = self.__class__()
        if isinstance(a, Variable1D):
            retval.L_sparse = self.L_sparse * a.L_sparse
        else:
            retval.L_sparse = self.L_sparse * a
        return retval

    def __rmul__(self, a):
        retval = self.__class__()
        if isinstance(a, Variable1D):
            retval.L_sparse = a.L_sparse * self.L_sparse
        else:
            retval.L_sparse = a * self.L_sparse
        return retval

    def __div__(self, a):
        retval = self.__class__()
        retval.L_sparse = self.L_sparse / a.L_sparse
        return retval

    def __iadd__(self, a):
        self.L_sparse += a.L_sparse
        return self

    def __isub__(self, a):
        self.L_sparse -= a.L_sparse
        return self

    def __imul__(self, a):
        self.L_sparse *= a.L_sparse
        return self

    def __idiv__(self, a):
        self.L_sparse /= a.L_sparse
        return self

    def __neg__(self):
        retval = self.__class__()
        retval.L_sparse = -self.L_sparse
        return retval

    def __pos__(self):
        retval = self.__class__()
        retval.L_sparse = self.L_sparse
        return retval



class OperatorDiff1D(_OperatorBase):
    """
    OperatorDiff1Derential operator class
    """
    def __init__(self, *args, **kwargs):
        _OperatorBase.__init__(self)

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
            
            print(" + src_grid.x_stencil_dofs: "+str(self.src_grid.x_stencil_dofs))
            print(" + src_grid.x_dofs: "+str(self.src_grid.x_dofs))
            print(" + dst_grid.x_stencil_dofs: "+str(self.dst_grid.x_stencil_dofs))
            print(" + dst_grid.x_dofs: "+str(self.dst_grid.x_dofs))
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
        
        if self.src_grid.boundaries[0].type == "periodic":
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
            if dst_x >= self.src_grid.domain_start and dst_x <= self.src_grid.x_dofs[0]:
                dst_x += self.src_grid.domain_size
                dst_x_domain_periodic_shifted = True
            
            """
            Finally, we need to fix a potentially cached src_idx_closest which
            is not correct anymore
            """
            if src_idx_closest >= self.src_grid.num_stencil_grid_points-2:
                src_idx_closest = 0
        
        
        """
        Given the current point on the dst variable,
        search for closest point on src variable
        """
        src_idx_closest_dist = np.inf
        for src_idx in range(src_idx_closest, self.src_grid.num_stencil_grid_points):
            """
            We use x_stencil_dofs here since this also includes the
            nodes for the Neumann and Dirichlet BC which will be
            "extracted" handled later on. This strongly simplifies things. 
            """
            
            src_x = self.src_grid.x_stencil_dofs[src_idx]
            
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
            if self.src_grid is self.dst_grid:
                assert self.src_grid.x_stencil_dofs[src_idx_closest] == dst_x, "There seems to be a problem with the coordinates"
        
        aligned_grid_point = (src_idx_closest_dist <= self.aligned_eps)
        
        if 0:
            print("*"*80)
            print(" + src_grid.x_stencil_dofs: "+str(self.src_grid.x_stencil_dofs))
            print(" + dst_grid.x_stencil_dofs: "+str(self.dst_grid.x_stencil_dofs))
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
                    if self.src_grid.boundaries[0].type in ["dirichlet", "neumann_extrapolated"] and self.dst_grid.boundaries[0].type in ["symmetric"]:
                        #if self.src_grid.x_stencil_dofs[0] > self.dst_grid.x_stencil_dofs[0]:
                        if True:
                            real_approx_order_equispaced = max(1, real_approx_order_equispaced-1)
                            real_approx_order_equispaced_postprocessed = True
                    
                if not real_approx_order_equispaced_postprocessed: 
                    if self.src_grid.boundaries[1].type in ["dirichlet", "neumann_extrapolated"] and self.dst_grid.boundaries[1].type in ["symmetric"]:
                        #if self.src_grid.x_stencil_dofs[-1] < self.dst_grid.x_stencil_dofs[-1]:
                        if True:
                            real_approx_order_equispaced = max(1, real_approx_order_equispaced-1)
                            real_approx_order_equispaced_postprocessed = True
            
            if not real_approx_order_equispaced_postprocessed:
                
                if self.src_grid.boundaries[0].type in ["periodic", "symmetric"] and self.src_grid.boundaries[1].type in ["periodic", "symmetric"]:
                #if self.src_grid.boundaries[0].type in ["periodic"] and self.src_grid.boundaries[1].type in ["periodic"]:
                    if diff_order % 2 == 0:
                        real_approx_order_equispaced += 1
                
                else:
                    """
                    TODO: Figure out why that's the case 
                    """
                    if self.src_grid.staggered and self.dst_grid.staggered:
                        pass
                    
                    
                    elif diff_order <= 3:
                    #if diff_order <= 3 and min_approx_order == 1:
                        real_approx_order_equispaced = max(2, real_approx_order_equispaced)
                    
                    """
                    if self.src_grid.boundaries[0].type in ["dirichlet0"] and self.src_grid.boundaries[1].type in ["dirichlet0"]:
                        
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
                    if self.src_grid.boundaries[0].type in ["dirichlet", "neumann_extrapolated"] and self.dst_grid.boundaries[0].type in ["symmetric"]:
                        if self.src_grid.x_stencil_dofs[0] > self.dst_grid.x_stencil_dofs[0]:
                            real_approx_order_equispaced = max(1, real_approx_order_equispaced-1)
                            real_approx_order_equispaced_postprocessed = True
                    
                if not real_approx_order_equispaced_postprocessed: 
                    if self.src_grid.boundaries[1].type in ["dirichlet", "neumann_extrapolated"] and self.dst_grid.boundaries[1].type in ["symmetric"]:
                        if self.src_grid.x_stencil_dofs[-1] < self.dst_grid.x_stencil_dofs[-1]:
                            real_approx_order_equispaced = max(1, real_approx_order_equispaced-1)
                            real_approx_order_equispaced_postprocessed = True
            
            if not real_approx_order_equispaced_postprocessed:
                if self.src_grid.boundaries[0].type in ["periodic", "symmetric"] and self.src_grid.boundaries[1].type in ["periodic", "symmetric"]:
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
            assert self.src_grid.num_dofs > num_stencil_dofs
            assert self.dst_grid.num_dofs > num_stencil_dofs
        
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
            if dst_x < self.src_grid.x_stencil_dofs[0] - self.aligned_eps:
                """
                Handle special case for symmetric BC on the left side
                """
                assert self.src_grid.boundaries[0].type == "symmetric", "This case should only exist for symmetric cases"
                
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
        if self.src_grid.boundaries[0].type == "periodic":
            
            src_range_wrapped = src_range % self.src_grid.num_dofs
            src_x_dofs = self.src_grid.x_dofs[src_range_wrapped]
            
            range_dist = self.src_grid.domain_end - self.src_grid.domain_start
            
            # Fix wrong x coordinates due to wrapping
            # e.g. -0.2 will be changed to -0.2+range_dist
            src_x_dofs -= (src_range < 0).astype(int)*range_dist
            src_x_dofs += (src_range >= self.src_grid.num_dofs).astype(int)*range_dist
            
            src_range = src_range_wrapped
            assert np.all((src_x_dofs[1:] - src_x_dofs[:-1]) > 0)
            
            bc_handled = True
        
        
        if not bc_handled and self.src_grid.boundaries[0].type == "symmetric":
                        
            if src_range[0] < 0:
                
                """
                Generate flag fields for parts which can be directly used and non-symmetric parts
                """
                flat_nosym = (src_range < 0).astype(int)
                flat_direct = (src_range >= 0).astype(int)
                
                if self.src_grid.boundaries[0].flip_sign:
                    stencil_factors = -1*flat_nosym + flat_direct
                    stencil_factors = stencil_factors.astype(float)
                
                if not self.src_grid.staggered:
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
                axis_x = self.src_grid.domain_start
                src_x_dofs = (2.0*axis_x - self.src_grid.x_stencil_dofs[src_range_symmetric])*flat_nosym + self.src_grid.x_stencil_dofs[src_range_symmetric]*flat_direct
                
                if 0:
                    print("x_stencil_dofs: "+str(self.src_grid.x_stencil_dofs))
                    print("+"*80)
                    print("src_range: "+str(src_range))
                    print("src_range_symmetric: "+str(src_range_symmetric))
                    print("src_x_dofs: "+str(src_x_dofs))
                    print("+"*80)
                    #raise Exception("TODO")
                
                src_range = src_range_symmetric
                
                bc_handled = True

        
        if not bc_handled and self.src_grid.boundaries[1].type == "symmetric":
            
            if src_range[-1] >= self.src_grid.num_stencil_grid_points:
                
                """
                Generate flag fields for parts which can be directly used and non-symmetric parts
                """
                flat_direct = (src_range < self.src_grid.num_stencil_grid_points).astype(int)
                flat_nosym = (src_range >= self.src_grid.num_stencil_grid_points).astype(int)
                
                if self.src_grid.boundaries[1].flip_sign:
                    stencil_factors = -1*flat_nosym + flat_direct
                    stencil_factors = stencil_factors.astype(float)
                
                if not self.src_grid.staggered:
                    """
                    DoF indices
                    """
                    axis_idx = self.src_grid.num_stencil_grid_points - 1
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
                    axis_idx = self.src_grid.num_stencil_grid_points - 1
                    src_range_symmetric = (2*axis_idx - src_range + 1)*flat_nosym + src_range*flat_direct
                    assert np.all(src_range_symmetric >= 0)
                
                """
                DoF coordinates
                """
                axis_x = self.src_grid.domain_end
                #src_x_dofs = (2.0*axis_x - self.src_grid.x_dofs[src_range_symmetric])*flat_nosym + self.src_grid.x_dofs[src_range_symmetric]*flat_direct
                src_x_dofs = (2.0*axis_x - self.src_grid.x_stencil_dofs[src_range_symmetric])*flat_nosym + self.src_grid.x_stencil_dofs[src_range_symmetric]*flat_direct


                if 0:
                    print("+"*80)
                    print("self.src_grid.num_dofs: "+str(self.src_grid.num_dofs))
                    print("src_range: "+str(src_range))
                    print("src_range_symmetric: "+str(src_range_symmetric))
                    print("src_x_dofs: "+str(src_x_dofs))
                    print("+"*80)
                    #raise Exception("TODO")
                
                src_range = src_range_symmetric
                
                bc_handled = True
        


        if self.src_grid.boundaries[0].type == "dirichlet" or self.src_grid.boundaries[0].type == "neumann_extrapolated":
            assert self.src_grid.x_stencil_dofs[1] == self.src_grid.x_dofs[0], "Internal Error"
        
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
            if src_range[-1] >= len(self.src_grid.x_stencil_dofs):
                s = src_range[-1] - len(self.src_grid.x_stencil_dofs) + 1
                src_range = [i-s for i in src_range]
                src_idx_shifted = True
            
            bc_handled = True
            
            src_x_dofs = self.src_grid.x_stencil_dofs[src_range]
            
            assert src_range[0] >= 0, "Negative index in src_range"
            assert src_range[-1] < self.src_grid.num_dofs+2, "Index too large in src_range"
        
        
        if not bc_handled:
            raise Exception("INTERNAL ERROR: Boundary condition not handled!")

        if 0:
            print("*"*80)
            print("diff_order: "+str(diff_order))
            print("num_stencil_default_offset: "+str(num_stencil_default_offset))
            print("dst_grid.num_dofs: "+str(self.dst_grid.num_dofs))
            print("dst_x: "+str(dst_x))
            print("self.src_grid.num_dofs: "+str(self.src_grid.num_dofs))
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
            src_x_dofs -= self.src_grid.domain_size
            if 0:
                print("src_x_dofs: "+str(src_x_dofs))
        
        
        assert not np.any((src_x_dofs[1:] - src_x_dofs[:-1]) < 0), "Internal error"
        
        return src_range, src_x_dofs, src_idx_closest, stencil_factors

    
    def setup(
        self,
        diff_order,
        min_approx_order,
        dst_grid,
        src_grid = None,
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
        
        if src_grid is None:
            src_grid = dst_grid
        
        if name is None:
            name = "OperatorDiff1D"+str(diff_order)+"_"+str(min_approx_order)
        self.name = name

        self.src_grid = src_grid
        self.dst_grid = dst_grid
        
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
            print("src_grid.x_stencil_dofs: "+str(self.src_grid.x_stencil_dofs))
            print("dst_grid.x_stencil_dofs: "+str(self.dst_grid.x_stencil_dofs))
            print("src_grid: "+str(self.src_grid.x_dofs))
            print("dst_grid: "+str(self.dst_grid.x_dofs))
            print("*"*80)
        
        
        """
        Allocate temporary storage.
        Note, that all src grid points are included here and that boundary conditions are coped with later on.
        This makes it more flexible.
        """
        L_sparse_tmp = libpdefd_matrix.MatrixSparseSetup((dst_grid.num_dofs, src_grid.num_stencil_grid_points))
        
        
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
                L_sparse_tmp[dst_idx, src_range[i]] += stencil[i]
            
            
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
                        assert np.isclose(0, np.sum(L_sparse_tmp[0,:].toarray()))
                        L_sparse_tmp[0,:] = 0
        
        if dst_x == src_grid.domain_end:
            if src_grid.boundaries[1].type == "symmetric":
                if not src_grid.boundaries[1].flip_sign:
                    if diff_order == 1:
                        assert np.isclose(0, np.sum(L_sparse_tmp[-1,:].toarray()))
                        L_sparse_tmp[-1,:] = 0
        
        
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
        self.c = np.zeros(dst_grid.num_dofs)
        
        if 0:
            np.set_printoptions(linewidth=180)
            print(L_sparse_tmp.toarray())
            print(self.c)
            print(src_grid.boundaries[0].dirichlet_value)
            print("*"*80)

        """
        Postprocessing of boundaries
        """
        if src_grid.boundaries[0].type == "periodic":

            """
            We only have to get rid of the last column
            """
            L_sparse_tmp = L_sparse_tmp[:,:src_grid.num_dofs]

        else:
            """
            First, handle boundary conditions at the beginning of the domain
            """
            if src_grid.boundaries[0].type == "dirichlet":
                # Getting a column as an array is a little bit more tricky with sp.sparse
                col = L_sparse_tmp[:,0].toarray()[:,0]
                self.c += col*src_grid.boundaries[0].dirichlet_value
                L_sparse_tmp = L_sparse_tmp[:,1:]
            
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
                    d = L_sparse_tmp[j,0]/stencil[0]
                    self.c[j] += d * neumann_value
                    
                    for i in range(1, len(stencil)):
                        L_sparse_tmp[j,i] -= d*stencil[i]
                
                L_sparse_tmp = L_sparse_tmp[:,1:]
            
            
            elif src_grid.boundaries[0].type == "symmetric":
                # Nothing to do
                pass
            
            
            else:
                raise Exception("Boundary condition '"+src_grid.boundaries[0].type+"' not supported")
            
            
            
            """
            Second, handle boundary conditions at the end of the domain
            """
            if src_grid.boundaries[1].type == "dirichlet":
                col = L_sparse_tmp[:,-1].toarray()[:,0]
                self.c += col*src_grid.boundaries[1].dirichlet_value
                
                L_sparse_tmp = L_sparse_tmp[:,:-1]
            
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
                    d = L_sparse_tmp[j,-1]/stencil[-1]
                    self.c[j] += d * neumann_value
                    
                    for i in range(0, len(stencil)-1):
                        L_sparse_tmp[j,i-len(stencil)] -= d*stencil[i]
                
                L_sparse_tmp = L_sparse_tmp[:,:-1]
            
            
            elif src_grid.boundaries[1].type == "symmetric":
                # Nothing to do
                pass
            
            
            else:
                raise Exception("Boundary condition '"+src_grid.boundaries[1].type+"' not supported")
        
        self.L_sparse = libpdefd_matrix.MatrixSparseCompute(L_sparse_tmp)
        
        
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
                        if not self.src_grid.staggered:
                            self.all_dst_dof_points_aligned = False
                    
                elif src_grid.boundaries[i].type in ['dirichlet']:
                    pass
                
                elif src_grid.boundaries[i].type in ['symmetric']:
                    pass
            
                else:
                    raise Exception("Case not handled")
        
        
        if 0:
            np.set_printoptions(linewidth=180)
            print(L_sparse_tmp.toarray())
            print(self.L_sparse.toarray())
            print(self.c)
        
        # Estimate possible cancellation errors
        self.L_sparse_min = self.L_sparse.min()
        self.L_sparse_max = self.L_sparse.max()
        self.L_sparse_cancellation_error = self.L_sparse_max - self.L_sparse_min
        
        if 0:
            np.set_printoptions(linewidth=180)
            print(self.L_sparse.toarray().shape)
            print(self.L_sparse.toarray())
            print(self.c)
        
        assert self.L_sparse.shape[0] == self.c.shape[0], "Internal Error"
        assert self.L_sparse.shape[0] == dst_grid.num_dofs, "Internal Error"
        


    def __str__(self):
        retstr = ""
        retstr += "Operator matrix:\n"
        retstr += str(self.get_L())
        return retstr



class OperatorDiffND(_OperatorBase):
    def __init__(self, *args, **kwargs):
        _OperatorBase.__init__(self)

        self.setup(*args, **kwargs)
    
    def setup(
        self,
        diff_dim: int,
        diff_order : int,
        min_approx_order : int,
        dst_grid : GridInfoND,
        src_grid  : GridInfoND = None,
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
        
        self.dim = diff_dim
        self.diff_order = diff_order
        self.min_approx_order = min_approx_order
        self.dst_grid = dst_grid
        self.src_grid = src_grid
        
        assert len(dst_grid) == len(src_grid)
        assert isinstance(dst_grid, GridInfoND)
        assert isinstance(src_grid, GridInfoND)
        
        num_dims = len(dst_grid)
        
        def get_bcast_matrix_L(L, i_dim):
            
            
            """
            Generate matrix with the linear operator 'L' in it
            """
            retm = libpdefd_matrix.MatrixSparseCompute(np.array([1]))
            
            for i in range(0, i_dim):
                M = libpdefd_matrix.eye(dst_grid.shape[i])
                #retm = sparse.kron(retm, M)
                retm = retm.kron(M)
                
            
            #retm = sparse.kron(retm, L)
            retm = retm.kron(L)
            
            for i in range(i_dim+1, num_dims):
                M = libpdefd_matrix.eye(src_grid.shape[i])
                #retm = sparse.kron(retm, M)
                retm = retm.kron(M)

            return retm
        
        
        def get_bcast_matrix_c(c, i_dim):
            
            """
            Generate matrix with which we can bcast the 'c' boundary conditions
            """
            retm = np.array([1])
            
            for i in range(0, i_dim):
                M = np.ones(dst_grid.shape[i])
                retm = np.kron(retm, M)
                
            retm = np.kron(retm, c)
            
            for i in range(i_dim+1, num_dims):
                M = np.ones(src_grid.shape[i])
                retm = np.kron(retm, M)
            
            return retm
        
        
        total_src_N = np.prod(src_grid.shape)
        total_dst_N = np.prod(dst_grid.shape)
        
        self.L_sparse = sparse.eye(total_src_N)
        self.c = np.zeros(total_src_N)
        
        if 0:
            print("self.L_sparse: "+str(self.L_sparse.toarray().shape))
            print("self.c: "+str(self.c.shape))
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
                    diff_order = self.diff_order if dim == diff_dim else 0,        # Differentiation or interpolation?
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
                    err = abs(diff_op.L_sparse - sparse.eye(np.product(dst_grid[dim].num_dofs))).max()
                    if err > 1e-10*(abs(diff_op.L_sparse).max()):
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
            BL = get_bcast_matrix_L(diff_op.L_sparse, dim)
            Bc = get_bcast_matrix_c(diff_op.c, dim)
                        
            if 0:
                print("diff_op.L_sparse", diff_op.L_sparse.toarray().shape)
                print("diff_op.c", diff_op.c.shape)
                print("BL", BL.toarray().shape)
                print("Bc", Bc.shape)
                print("self.c", self.c.shape)
            
            if dim == 0:
                self.L_sparse = BL
                self.c = Bc
            else:
                self.L_sparse = BL.dot(self.L_sparse)
                self.c = BL.dot(self.c) + Bc
            
            if self.L_sparse.shape[0] != self.c.shape[0]:
                print("Internal Error")
                print(" + dim: "+str(dim))
                print(" + self.L_sparse.shape: "+str(self.L_sparse.shape))
                print(" + self.c.shape: "+str(self.c.shape))
                raise Exception("Internal Error")
            
            if 0:
                print("INFO")
                print(" + dim: "+str(dim))
                print(" + self.L_sparse.shape: "+str(self.L_sparse.shape))
                print(" + self.c.shape: "+str(self.c.shape))
                print(" + dst_grid.shape: "+str(dst_grid.shape))
                
            ndofs = np.prod(dst_grid.shape)
            assert self.L_sparse.shape[0] == self.c.shape[0], "Internal Error"
        
        
        assert self.L_sparse.shape[0] == self.c.shape[0], "Internal Error"
        assert self.L_sparse.shape[0] == np.prod(dst_grid.shape), "Internal Error"

