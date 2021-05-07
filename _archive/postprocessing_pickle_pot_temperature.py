#! /usr/bin/env python

import sys
import pickle
import argparse

import libpdefd



parser = argparse.ArgumentParser()

parser.add_argument('--input-temperature', dest="input_t_filename", type=str, help="Pickle temperature")
parser.add_argument('--input-pressure', dest="input_p_filename", type=str, help="Pickle pressure")
parser.add_argument('--output', dest="output", type=str, help="Output filename")
parser.add_argument('--dpi', dest="dpi", type=int, help="DPI in case of bitmap output format")


args = parser.parse_args()

if args.input_t_filename == None:
    raise Exception("Please provide at least --input-temperature")

if args.input_p_filename == None:
    raise Exception("Please provide at least --input-pressure")

if args.input_t_filename == args.output:
    raise Exception("Input and output are the same")

if args.input_p_filename == args.output:
    raise Exception("Input and output are the same")



print("*"*80)
print("Input temperature: "+str(args.input_t_filename))
print("Input pressure: "+str(args.input_p_filename))
print("Output: "+str(args.output))
print("*"*80)


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

file_p = open(args.input_t_filename, 'rb')
pickle_data_p  = pickle.load(file_p)

data_p = pickle_data_p['var_data']


file_t = open(args.input_p_filename, 'rb')
pickle_data_t  = pickle.load(file_t)

data_t = pickle_data_t['var_data']


simconfig = pickle_data_p['simconfig']
pickle_data = pickle_data_p


"""
Compute Exner pressure
"""
def exner(data_p):
    alpha = simconfig.const_R / simconfig.const_c_p
    return np.power(data_p / simconfig.const_p0, alpha)


data_pot_t = data_t * exner(data_p)

var_name = pickle_data['var_name']


if 1:
    def plot_update_title(i, vis_variable_name, title_prefix=""):
        title = title_prefix
        title += vis_variable_name
        #title += ", t="+str(round(i*dt/(60*60), 3))+" h"
        title += ", t="+str(round(i*pickle_data['state_dt'], 3))+" sec"
        vis.set_title(title)

    vis = libpdefd.vis.Visualization2DMesh(
        vis_dim_x = pickle_data['simconfig'].vis_dim_x,
        vis_dim_y = pickle_data['simconfig'].vis_dim_y,
        vis_slice = pickle_data['simconfig'].vis_slice,
        rescale = 1.0
    )

    vis.update_plots(pickle_data['var_gridinfo'], pickle_data['var_data'])
    
    plot_update_title(pickle_data['state_num_timestep'], var_name)

    if args.output == None:
        vis.show()
    else:
        if args.dpi == None:
            vis.savefig(args.output)
        else:
            vis.savefig(args.output, dpi=args.dpi)


