import numpy as np
import matplotlib.pyplot as plt
import libpdefd.plot_config as pc
import libpdefd
import sys



class _VisualizationBase:
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        pass
    
    def set_title(self, title):
        self.ax.set_title(title)


    def show(self, **kwargs):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.show(**kwargs)

    
class Visualization1D(_VisualizationBase):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        _VisualizationBase.__init__(self)
        self.setup(*args, **kwargs)


        
    def setup(
            self,
            use_symlog = False,
    ):
        self.use_symlog = use_symlog
        self.firsttime = True
        
        self.fig, self.ax = pc.setup()
        self.ps = pc.PlotStyles()
        
        
    def _update_plots_firsttime(
            self,
            grids_list : list,
            variables_list : list,
    ):
        assert(len(grids_list) == len(variables_list))
        
        maxy = -1

        self.lines = [None for i in range(len(grids_list))]
        for i in range(len(grids_list)):
            grid = grids_list[i]
            variable = variables_list[i]
            
            if isinstance(grid, libpdefd.GridND):
                grid1d = grid[0]
            else:
                grid1d = grid

            plotstyle = self.ps.getNextStyle(len(variable.data), 15)
            self.lines[i], = self.ax.plot(grid1d.x_dofs, variable.data, **plotstyle, label="u(x)")
        
            maxy = np.max([maxy, np.max(np.abs(variable.data))])
        
        self.ax.legend()
        self.ax.set_ylim(-maxy, maxy)
        if self.use_symlog:
            self.ax.set_yscale("symlog", linthresh=1e-4)
        
        
    def update_plots(
            self,
            grids_list,
            variables_list
    ):
        if not isinstance(variables_list, list):
            variables_list = [variables_list]
            
        if not isinstance(grids_list, list):
            grids_list = [grids_list]

        if self.firsttime:
            self._update_plots_firsttime(grids_list, variables_list)
            self.firsttime = False
            return

        for i in range(len(grids_list)):
            self.lines[i].set_ydata(variables_list[i].data)



class Visualization2DMesh(_VisualizationBase):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        _VisualizationBase.__init__(self)
        self.setup(*args, **kwargs)


        
    def setup(
            self,
            vis_dim_x = 0,
            vis_dim_y = 1,
            vis_slice : list = [],
            rescale = 1,
            use_symlog = False
    ):
        """
        Prepare plotting
        """
        self.vis_dim_x = vis_dim_x
        self.vis_dim_y = vis_dim_y
        self.vis_slice = vis_slice
        self.rescale = rescale
        self.use_symlog = use_symlog
        
        self.firsttime = True
        
        self.fig, self.ax = pc.setup()
        self.ps = pc.PlotStyles()
        
        
    def get_dimreduced_data(
            self,
            data
    ):
        
        slices = [slice(0, data.shape[i]) for i in range(len(data.shape))]
        
        for i in range(len(data.shape)):
            if i == self.vis_dim_x or i == self.vis_dim_y:
                continue
            
            slices[i] = self.vis_slice[i]

        data = data[tuple(slices)]
        
        if self.vis_dim_x < self.vis_dim_y:
            data = data.transpose()
        
        return data
    
    
    def _update_plots_firsttime(
            self,
            grid_info_nd : libpdefd.GridInfoND,
            variable
    ):
        if isinstance(variable, libpdefd.VariableND):
            data = self.get_dimreduced_data(variable.data)
        else:
            data = variable

        if 0:
            self.maxy = np.max(np.abs(data))
            if self.maxy <= 1e-12:
                self.maxy = 1e-12
                
            vmin = -self.maxy*self.rescale
            vmax = self.maxy*self.rescale
            
        else:
            self.maxy = np.max(data)
            self.miny = np.min(data)
            
            if self.maxy-self.miny <= 1e-12:
                self.maxy += 1e-12
                self.miny -= 1e-12

            vmin = self.miny*self.rescale
            vmax = self.maxy*self.rescale
        
        mesh_grid_coords = grid_info_nd.get_meshcoords_staggered()
        
        dx = grid_info_nd.grids1d_list[0].domain_end - grid_info_nd.grids1d_list[0].domain_start
        dy = grid_info_nd.grids1d_list[1].domain_end - grid_info_nd.grids1d_list[1].domain_start
        
        aspect = dy/dx*data.shape[1]/data.shape[0]
        
        if 0:
            self.colormesh = plt.pcolormesh(
                mesh_grid_coords[self.vis_dim_x],
                mesh_grid_coords[self.vis_dim_y],
                data,
                vmin = vmin,
                vmax = vmax,
                cmap = "viridis"
            )

            self.ax.set_aspect(1.0)
        
        else:
            self.colormesh = plt.imshow(
                np.flip(data, axis=0),
                vmin = vmin,
                vmax = vmax,
                cmap = "viridis",
                aspect = aspect
            )
        #print(aspect)
        
        self.ax.set_xlabel("D"+str(self.vis_dim_x))
        self.ax.set_ylabel("D"+str(self.vis_dim_y))
        
        self.cbar = self.fig.colorbar(self.colormesh)
        #self.cbar.set_label('rho variable')
        
        self.fig.tight_layout(rect=[0,0,1,.9])
    
    
    
    def update_plots(
            self,
            grid_info_nd,
            variable_data
    ):
        if self.firsttime:
            self._update_plots_firsttime(grid_info_nd, variable_data)
            self.firsttime = False
            return
        
        # Update limits
        data = self.get_dimreduced_data(variable_data)
        maxy = np.max(np.abs(data))
        
        if maxy > self.maxy or True:
            self.cbar.remove()
            self.colormesh.remove()
            self._update_plots_firsttime(grid_info_nd, variable_data)
            return
            
        self.colormesh.set_array(variable_data)


    def savefig(self, filename, **kwargs):
        self.fig.savefig(filename, **kwargs)
        
        
