#! /usr/bin/env python

import sys
import numpy as np
import pickle
import argparse

import libpdefd



parser = argparse.ArgumentParser()

parser.add_argument('--figscale', dest="figscale", type=float, help="Scaling of figure")
parser.add_argument('--filename', dest="filename", type=str, help="Pickle filename")
parser.add_argument('--output', dest="output", type=str, help="Output filename")
parser.add_argument('--dpi', dest="dpi", type=int, help="DPI in case of bitmap output format")


args = parser.parse_args()
if args.figscale is None:
    args.figscale = 1.0

if args.filename == None:
    raise Exception("Please provide at least --filename")

#if args.output == None:
#    args.output = args.filename.replace(".pickle", ".pdf")

if args.filename == args.output:
    raise Exception("Input and output are the same")



print("*"*80)
print("Input: "+str(args.filename))
print("Output: "+str(args.output))
print("*"*80)

with open(args.filename, 'rb') as file:
    pickle_data  = pickle.load(file)
    
    """
    pickle_data = {
        'var_name': var_name,
        'var_data': var_data,
        'var_gridinfo': var_gridinfo,
        'state_dt': dt,
        'state_num_timestep': num_timestep,
        'state_simtime': simtime,
        'simconfig': simconfig,
    }
    """
    
    var_name = pickle_data['var_name']
    
    def plot_update_title(i, vis_variable_name, title_prefix=""):
        title = title_prefix
        title += vis_variable_name
        #title += ", t="+str(round(i*dt/(60*60), 3))+" h"
        title += ", t="+str(round(i*pickle_data['state_dt'], 3))+" sec"
        vis.set_title(title)

    vis = libpdefd.visualization.Visualization2DMesh(
        vis_dim_x = pickle_data['simconfig'].vis_dim_x,
        vis_dim_y = pickle_data['simconfig'].vis_dim_y,
        vis_slice = pickle_data['simconfig'].vis_slice,
        rescale = 1.0,
        figscale = args.figscale,
    )
    

    if var_name in ['pot_t', 'pot_t_diff']:
        
        contour_levels = np.arange(-100, 101, 1)
        
        # Remove contour around 0
        contour_levels = np.delete(contour_levels, np.where(np.isclose(contour_levels, 0)))
        
    else:
        contour_levels = None
    
    vis.update_plots(pickle_data['var_gridinfo'], pickle_data['var_data'], contour_levels = contour_levels)
    
    plot_update_title(pickle_data['state_num_timestep'], var_name)

    if args.output == None:
        vis.show()
    else:
        #kwargs = {'figsize': (1920, 1080)}
        kwargs = {}
        if args.dpi == None:
            vis.savefig(args.output, **kwargs)
        else:
            vis.savefig(args.output, dpi=args.dpi, **kwargs)


