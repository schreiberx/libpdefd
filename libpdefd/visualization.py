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
        use_symlog = False,
        figscale = None
    ):
        """
        Prepare plotting
        """
        self.vis_dim_x = vis_dim_x
        self.vis_dim_y = vis_dim_y
        self.vis_slice = vis_slice
        self.rescale = rescale
        self.use_symlog = use_symlog
        self.figscale = figscale
        
        self.firsttime = True
        
        self.fig, self.ax = pc.setup(scale=figscale)
        self.ps = pc.PlotStyles()
        
        """
        colormesh
        imshow
        """
        self.plot_mode = "colormesh"
        
        self.contour = None
        
    
    
    def get_dimreduced_data(
        self,
        variable_data
    ):
        assert isinstance(variable_data, np.ndarray)
        
        slices = [slice(0, variable_data.shape[i]) for i in range(len(variable_data.shape))]
        
        for i in range(len(variable_data.shape)-1, -1, -1):
            if i == self.vis_dim_x or i == self.vis_dim_y:
                continue
            
            slices[i] = self.vis_slice[i]
 
        variable_data = variable_data[tuple(slices)]
        
        # y for row index, x for col index
        variable_data = variable_data.transpose()
        
        #if self.vis_dim_x < self.vis_dim_y:
        #    data = data.transpose()
        #print(data.shape)
        
        return variable_data
    
    
    def _update_plots_firsttime(
        self,
        grid_info_nd : libpdefd.GridInfoND,
        variable_data : np.ndarray
    ):
        assert isinstance(grid_info_nd, libpdefd.GridInfoND)
        assert isinstance(variable_data, np.ndarray)
        
        if 1:
            self.maxabsy = np.max(np.abs(variable_data))
            
            
            self.maxy = np.max(variable_data)
            self.miny = np.min(variable_data)
            
            if self.maxy-self.miny <= 1e-12:
                self.maxy += 1e-12
                self.miny -= 1e-12
            
            vmin = self.miny*self.rescale
            vmax = self.maxy*self.rescale
        
        mesh_grid_coords = grid_info_nd.get_meshcoords_staggered()
        
        dx = grid_info_nd.grids1d_list[0].domain_end - grid_info_nd.grids1d_list[0].domain_start
        dy = grid_info_nd.grids1d_list[1].domain_end - grid_info_nd.grids1d_list[1].domain_start
        
        num_ticks = 4
        
        if self.plot_mode == "colormesh":
            self.plotobject = plt.pcolormesh(
                mesh_grid_coords[self.vis_dim_x],
                mesh_grid_coords[self.vis_dim_y],
                variable_data,
                vmin = vmin,
                vmax = vmax,
                cmap = "viridis"
            )

            self.ax.set_aspect(1.0)
            
            self.ax.set_xticks(np.linspace(mesh_grid_coords[self.vis_dim_x][0], mesh_grid_coords[self.vis_dim_x][-1], num_ticks, endpoint=True))
            self.ax.set_yticks(np.linspace(mesh_grid_coords[self.vis_dim_y][0], mesh_grid_coords[self.vis_dim_y][-1], num_ticks, endpoint=True))
        
        elif self.plot_mode == "imshow":
            aspect = dy/dx*variable_data.shape[1]/variable_data.shape[0]
            
            self.plotobject = plt.imshow(
                np.flip(variable_data, axis=0),
                vmin = vmin,
                vmax = vmax,
                cmap = "viridis",
                aspect = aspect
            )

            self.ax.set_xticks(np.linspace(0, variable_data.shape[1], num_ticks, endpoint=False))
            self.ax.set_yticks(np.linspace(0, variable_data.shape[0], num_ticks, endpoint=False))
            
        else:
            raise Exception("Plotting mode '"+self.plot_mode+"' unknown")
        
            
        xtickslabels = ["{:g}".format(i) for i in np.linspace(grid_info_nd.grids1d_list[0].domain_start, grid_info_nd.grids1d_list[0].domain_end, 4, endpoint=True)]
        self.ax.set_xticklabels(xtickslabels)
        
        ytickslabels = ["{:g}".format(i) for i in np.linspace(grid_info_nd.grids1d_list[1].domain_start, grid_info_nd.grids1d_list[1].domain_end, 4, endpoint=True)]
        self.ax.set_yticklabels(ytickslabels)
        
        self.ax.set_xlabel("D"+str(self.vis_dim_x))
        self.ax.set_ylabel("D"+str(self.vis_dim_y))
        
        self.cbar = self.fig.colorbar(self.plotobject)
        
        self.fig.tight_layout(rect=[0,0,1,.9])
    
    
    def update_plots(
        self,
        grid_info_nd : libpdefd.GridInfoND,
        variable_data,
        contour_levels = None
    ):
        if not isinstance(variable_data, np.ndarray):
            variable_data = variable_data.to_numpy_array()

        dim_reduced_data = self.get_dimreduced_data(variable_data)
        
        if self.firsttime:
            self._update_plots_firsttime(grid_info_nd, dim_reduced_data)
            self.firsttime = False
        
        else:
            # Update limits
            maxabsy = np.max(np.abs(dim_reduced_data))
            
            if maxabsy > self.maxabsy:
                self.cbar.remove()
                self.plotobject.remove()
                self._update_plots_firsttime(grid_info_nd, dim_reduced_data)
                self.maxabsy = maxabsy
            
            else:
                self.plotobject.set_array(dim_reduced_data)
        
        if contour_levels is not None:
            
            if self.plot_mode == "colormesh":
                x = grid_info_nd.grids1d_list[0].x_dofs
                y = grid_info_nd.grids1d_list[1].x_dofs
                
                X, Y = np.meshgrid(x, y)
                
            elif self.plot_mode == "imshow":
                raise Exception("Doesn't really work although it's much faster. Don't use this in combination with contours!")
                x = grid_info_nd.grids1d_list[0].x_dofs
                y = grid_info_nd.grids1d_list[1].x_dofs
                
                X, Y = np.meshgrid(x, y)
            
            
            if self.contour != None:
                for c in self.contour.collections: 
                    c.remove()
            
            self.contour = self.ax.contour(X, Y, dim_reduced_data, levels=contour_levels, colors='black', linewidths=0.25)
            #self.ax.clabel(self.contour, inline=True, fontsize=3, colors="black")
    
    
    def savefig(self, filename, **kwargs):
        self.fig.savefig(filename, **kwargs)

