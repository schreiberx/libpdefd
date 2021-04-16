import numpy as np
from libpdefd.core.boundary import *


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
    
        self.shape = (self.num_dofs, )
    
    
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

