#! /usr/bin/env python3


import numpy as np
import scipy as sp
import scipy.sparse as sparse
import itertools
import sys
import libpdefd
import argparse


boundary_src_ = ['dirichlet', 'neumann_extrapolated', 'symmetric', 'periodic']
boundary_dst_ = ['symmetric', 'periodic', 'dirichlet', 'neumann_extrapolated']

boundary_src_right_ = ["", "neumann_extrapolated", "symmetric"]
boundary_dst_right_ = ["", "neumann_extrapolated", "symmetric"]
#boundary_src_right_ = [""]
#boundary_dst_right_ = [""]

grid_src_type_ = ["auto", "homemade_nonequidistant", "homemade_equidistant"]
grid_dst_type_ = ["auto", "homemade_nonequidistant", "homemade_equidistant"]

staggered_src_grid_ = [False, True]
staggered_dst_grid_ = [False, True]

diff_order_ = [0, 1, 2, 3, 4, 5]
min_approx_order_ = [1, 2, 3, 4, 5]
plot_solution = False
plot_stencil = None

check_errors = True


if len(sys.argv) > 1:
    if sys.argv[1] == "print_parallel_jobs":
        
        """
        Print all variants of parameters to directly call test jobs to command line
        """
        for (boundary_src, boundary_dst, boundary_src_right, boundary_dst_right) in itertools.product(boundary_src_, boundary_dst_, boundary_src_right_, boundary_dst_right_):
            
            if boundary_src_right == "":
                """
                Skip if this combination is explicitly available
                """
                if boundary_src in boundary_src_ and boundary_src in boundary_src_right_:
                    continue
                boundary_src_right = boundary_src
            
            if boundary_dst_right == "":
                """
                Skip if this combination is explicitly available
                """
                if boundary_dst in boundary_dst_ and boundary_dst in boundary_dst_right_:
                    continue
                boundary_dst_right = boundary_dst
            
            """
            Skip if periodicity doesn't match
            """
            if boundary_src == "periodic":
                if boundary_src_right != "periodic":
                    continue
                if boundary_dst != "periodic":
                    continue
                if boundary_dst_right != "periodic":
                    continue
            
            if boundary_dst == "periodic":
                if boundary_dst_right != "periodic":
                    continue
                if boundary_src != "periodic":
                    continue
                if boundary_src_right != "periodic":
                    continue
            
            for (grid_src_type, grid_dst_type, staggered_src_grid, staggered_dst_grid) in itertools.product(grid_src_type_, grid_dst_type_, staggered_src_grid_, staggered_dst_grid_):
                for (diff_order, min_approx_order) in itertools.product(diff_order_, min_approx_order_):
                    
                    cmd_line = "python "+sys.argv[0]
                    cmd_line += " --boundary-src="+boundary_src
                    cmd_line += " --boundary-src-right="+boundary_src_right
                    cmd_line += " --boundary-dst="+boundary_dst
                    cmd_line += " --boundary-dst-right="+boundary_dst_right
                    cmd_line += " --grid-src-type="+grid_src_type
                    cmd_line += " --grid-dst-type="+grid_src_type
                    cmd_line += " --staggered-src-grid="+str(staggered_src_grid)
                    cmd_line += " --staggered-dst-grid="+str(staggered_dst_grid)
                    cmd_line += " --diff-order="+str(diff_order)
                    cmd_line += " --min-approx-order="+str(min_approx_order)
                    print(cmd_line)
        
        sys.exit(0)



start_res2 = 5
end_res2 = 13

def tobool(s):
    return s.lower() in ['true', '1']


if 1:
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundary-src', dest="boundary_src", type=str, help="Boundaries for source field, default: "+(",".join(boundary_src_)))
    parser.add_argument('--boundary-dst', dest="boundary_dst", type=str, help="Boundaries for destination field, default: "+(",".join(boundary_dst_)))
    parser.add_argument('--boundary-src-right', dest="boundary_src_right", type=str, help="Boundaries for right source field, default: "+(",".join(boundary_src_right_)))
    parser.add_argument('--boundary-dst-right', dest="boundary_dst_right", type=str, help="Boundaries for right destination field, default: "+(",".join(boundary_dst_right_)))
    parser.add_argument('--grid-src-type', dest="grid_src_type", type=str, help="Grid type, default: "+(",".join(grid_src_type_)))
    parser.add_argument('--grid-dst-type', dest="grid_dst_type", type=str, help="Grid type, default: "+(",".join(grid_dst_type_)))
    parser.add_argument('--staggered-src-grid', dest="staggered_src_grid", type=str, help="Src. grid staggering, default: "+(",".join([str(i) for i in staggered_src_grid_])))
    parser.add_argument('--staggered-dst-grid', dest="staggered_dst_grid", type=str, help="Dst. grid staggering, default: "+(",".join([str(i) for i in staggered_dst_grid_])))
    parser.add_argument('--diff-order', dest="diff_order", type=str, help="Diff order, default: "+(",".join([str(i) for i in diff_order_])))
    parser.add_argument('--min-approx-order', dest="min_approx_order", type=str, help="Minimum approx. order, default: "+(",".join([str(i) for i in min_approx_order_])))
    parser.add_argument('--start-res2', dest="start_res2", type=int, help="Exponent of pow2 for start of range of resolutions: "+str(start_res2))
    parser.add_argument('--end-res2', dest="end_res2", type=int, help="Exponent of pow2 for end of range of resolutions: "+str(end_res2))
    parser.add_argument('--plot-solution', dest="plot_solution", type=str, help="Activate plotting: "+str(plot_solution))
    parser.add_argument('--plot-stencils', dest="plot_stencil", type=str, help="Activate plotting of stencil dependencies (plot, file, None): "+str(plot_stencil))
    parser.add_argument('--check-errors', dest="check_errors", type=str, help="Check errors: "+str(check_errors))
    args = parser.parse_args()
    
    if args.boundary_src != None:
        boundary_src_ = args.boundary_src.split(",")
    
    if args.boundary_dst != None:
        boundary_dst_ = args.boundary_dst.split(",")
        
    if args.boundary_src_right != None:
        boundary_src_right_ = args.boundary_src_right.split(",")
    
    if args.boundary_dst_right != None:
        boundary_dst_right_ = args.boundary_dst_right.split(",")
    
    if args.grid_src_type != None:
        grid_src_type_ = args.grid_src_type.split(",")
    
    if args.grid_dst_type != None:
        grid_dst_type_ = args.grid_dst_type.split(",")
    
    if args.staggered_src_grid != None:
        staggered_src_grid_ = [tobool(i) for i in args.staggered_src_grid.split(",")]
    
    if args.staggered_dst_grid != None:
        staggered_dst_grid_ = [tobool(i) for i in args.staggered_dst_grid.split(",")]
    
    if args.diff_order != None:
        diff_order_ = [int(i) for i in args.diff_order.split(",")]
    
    if args.min_approx_order != None:
        min_approx_order_ = [int(i) for i in args.min_approx_order.split(",")]
   
    if args.start_res2 != None:
        start_res2 = args.start_res2
        
    if args.end_res2 != None:
        end_res2 = args.end_res2
   
    if args.plot_solution != None:
        plot_solution = tobool(args.plot_solution)
        
    if args.plot_stencil != None:
        plot_stencil = args.plot_stencil
        
    if args.check_errors != None:
        check_errors = tobool(args.check_errors)



res_ = [2**i for i in range(start_res2, end_res2)]

neumann_diff_order = 2



if 0:
    domain_start = -100
    domain_end = 200
    
elif 1:
    domain_start = -16
    domain_end = 16
    
else:
    domain_start = 0
    domain_end = 1

domain_size = domain_end - domain_start



def test_fun_periodic(x, diff_order = 0, freq = None, shift = 1.3):
    """
    Periodic trigonometric function
        f(x) = sin(2*pi*x)
        f'(x) = 2*pi*cos(2*pi*x)

    """
    
    if freq == None:
        freq = 7.0
    
    # We need a higher frequency to avoid a very fast convergence
    retval = 1
    
    x = (x - domain_start)/domain_size
    retval *= (1.0/domain_size)**diff_order

    retval *= (2.0*np.pi*freq)**diff_order
    
    if diff_order % 2 == 0:
        retval *= np.sin(x*2.0*np.pi*freq+shift)
    else:
        retval *= np.cos(x*2.0*np.pi*freq+shift)
    
    if diff_order//2 % 2 == 1:
        retval *= -1
    
    if diff_order == 0:
        retval += 1.5

    return retval


def test_fun_nonperiodic(x, diff_order = 0, freq = None):

    retval1 = test_fun_periodic(x, diff_order, freq = freq)
    
    x = (x - domain_start)/domain_size

    retval2 = np.math.factorial(6)/np.math.factorial(6-diff_order)
    retval2 *= x**(6 - diff_order)

    retval2 *= (1.0/domain_size)**diff_order

    return retval1 + retval2


def test_fun_periodic_symmetric(x, diff_order = 0, freq = None):
    """
    Return function which is symmetric around boundaries
    """
    return test_fun_periodic(
        x,
        diff_order+1,
        shift = 0,
        freq = freq
    )


verbose = 1


for (boundary_src, boundary_dst, boundary_src_right, boundary_dst_right) in itertools.product(boundary_src_, boundary_dst_, boundary_src_right_, boundary_dst_right_):
    for (grid_src_type, grid_dst_type, staggered_src_grid, staggered_dst_grid) in itertools.product(grid_src_type_, grid_dst_type_, staggered_src_grid_, staggered_dst_grid_):
        for (diff_order, min_approx_order) in itertools.product(diff_order_, min_approx_order_):
    
            if boundary_src_right == "":
                boundary_src_right = boundary_src
            
            if boundary_dst_right == "":
                boundary_dst_right = boundary_dst
            
            
            if diff_order + min_approx_order > 8:
                continue
    
            error_norm = "max"
            #error_norm = "l2"
            
            freq = (diff_order+min_approx_order+1) 
            freq = 7.0

            if verbose > 0:
                print("")
                print("*"*80)
                print("Next test case")
                print(" + boundary_src: "+str(boundary_src))
                print(" + boundary_dst: "+str(boundary_dst))
                print(" + boundary_src_right: "+str(boundary_src_right))
                print(" + boundary_dst_right: "+str(boundary_dst_right))
                print(" + grid_src_type: "+str(grid_src_type))
                print(" + grid_dst_type: "+str(grid_dst_type))
                print(" + staggered_src_grid: "+str(staggered_src_grid))
                print(" + staggered_dst_grid: "+str(staggered_dst_grid))
                print(" + diff_order: "+str(diff_order))
                print(" + min_approx_order: "+str(min_approx_order))
                print(" + error_norm: "+str(error_norm))
                print(" + domain_start: "+str(domain_start))
                print(" + domain_end: "+str(domain_end))
                print(" + plot_solution: "+str(plot_solution))
                print(" + plot_stencil: "+str(plot_stencil))
                print(" + check_errors: "+str(check_errors))
                print(" + freq: "+str(freq))

                print("*"*80)
    

            def get_fun(boundary_type, boundary_type_right):
                if boundary_type == "periodic":
                    assert boundary_type_right == "periodic"
                    test_fun_ = test_fun_periodic
                    print(" + function: test_fun_periodic")
                
                elif boundary_type == "symmetric" or boundary_type_right == "symmetric":
                    test_fun_ = test_fun_periodic_symmetric
                    print(" + function: test_fun_periodic_symmetric")
                
                elif boundary_type == "dirichlet":
                    test_fun_ = test_fun_nonperiodic
                    print(" + function: test_fun_nonperiodic")
                
                elif boundary_type == "neumann_extrapolated":
                    test_fun_ = test_fun_nonperiodic
                    print(" + function: test_fun_nonperiodic")
                    
                else:
                    raise Exception("Boundary type '"+boundary_type+"' not supported")
                
                return test_fun_

            test_fun_ = get_fun(boundary_src, boundary_src_right)
            
            
            def test_fun(x, diff_order):
                return test_fun_(x, diff_order=diff_order, freq=freq)
            
            if boundary_src == "symmetric":
                assert np.isclose(test_fun(domain_start, 1), 0)
                
            if boundary_src_right == "symmetric":
                assert np.isclose(test_fun(domain_end, 1), 0)
            
            
            def get_bc(boundary_type, boundary_type_right, test_fun):
                
                if boundary_type == "periodic":
                    boundary_left = libpdefd.BoundaryPeriodic()
                
                elif boundary_type == "dirichlet":
                    boundary_left = libpdefd.BoundaryDirichlet(test_fun(domain_start, 0))
                
                elif boundary_type == "neumann_extrapolated":
                    boundary_left = libpdefd.BoundaryNeumannExtrapolated(test_fun(domain_start, diff_order=neumann_diff_order), diff_order=neumann_diff_order)
                    
                elif boundary_type == "symmetric":
                    boundary_left = libpdefd.BoundarySymmetric()
                    
                else:
                    raise Exception("Boundary type '"+boundary_type+"' not supported")


                if boundary_type_right == "periodic":
                    boundary_right = libpdefd.BoundaryPeriodic()
                
                elif boundary_type_right == "dirichlet":
                    boundary_right = libpdefd.BoundaryDirichlet(test_fun(domain_end, 0))
                
                elif boundary_type_right == "neumann_extrapolated":
                    boundary_right = libpdefd.BoundaryNeumannExtrapolated(test_fun(domain_end, diff_order=neumann_diff_order), diff_order=neumann_diff_order)
                    
                elif boundary_type_right == "symmetric":
                    boundary_right = libpdefd.BoundarySymmetric()
                    
                else:
                    raise Exception("Boundary type right '"+boundary_type_right+"' not supported")

                
                return boundary_left, boundary_right
            
    
            prev_error = 0
            conv_list = []
    
            for res in res_:
        
                # Resolution including all grid points, including boundaries
                N = res+1
        
                
                def get_grid(grid_type, staggered, name, boundaries):
                    grid = libpdefd.GridInfo1D(name=name)
                    
                    if grid_type == "auto":
                        grid.setup_autogrid(
                            domain_start,
                            domain_end,
                            regular_num_grid_points = N,
                            boundaries = boundaries,
                            staggered = staggered
                        )
                    
                    elif grid_type == "homemade_equidistant":
                        
                        if staggered:
                            if isinstance(boundaries[0], libpdefd.BoundaryPeriodic):
                                x = np.linspace(0, 1, N, endpoint=True) + 0.5/(N-1)
                                x *= domain_size
                                x += domain_start
                            
                            else:
                                xt = np.linspace(0, 1, N, endpoint=True) + 0.5/(N-1)
                                xt = xt[0:-1]
                                
                                x = np.empty(N+1)
                                x[1:-1] = xt
                                x[0] = 0
                                x[-1] = 1
                                
                                x *= domain_size
                                x += domain_start
                        
                        else:
                            x = np.linspace(domain_start, domain_end, N, endpoint=True)
                        
                        grid.setup_manualgrid(x, x_start=domain_start, x_end=domain_end, boundaries=boundaries, staggered=staggered)
                        
                    
                    elif grid_type == "homemade_nonequidistant":
                        
                        if staggered:
                            xt = np.linspace(0, 1, N, endpoint=True) + 0.5/(N-1)
                            xt = xt[0:-1]
                            
                            x = np.empty(N+1)
                            x[1:-1] = xt
                            x[0] = 0
                            x[-1] = 1
                            
                            x = np.tanh(x*2.0-1.0)
                            x /= np.abs(x[0])
                            
                            x = x*0.5+0.5
                            
                            x *= domain_size
                            x += domain_start
                            
                        else:
                            x = np.linspace(0, 1, N, endpoint=True)
                            
                            x = np.tanh(x*2.0-1.0)
                            x /= np.abs(x[0])
                            
                            x = x*0.5+0.5
                            
                            x *= domain_size
                            x += domain_start
                     
                        grid.setup_manualgrid(x, x_start=domain_start, x_end=domain_end, boundaries=boundaries, staggered=staggered)
                    
                    else:
                        raise Exception("GridInfo1D type "+str(grid_type)+" not supported")
                
                    return grid
                
                
                """
                Setup source grid
                """
                bc_src_left, bc_src_right = get_bc(boundary_src, boundary_src_right, test_fun)
                
                src_grid = get_grid(grid_src_type, staggered_src_grid, name="src_u", boundaries=[bc_src_left, bc_src_right])
                
                
                """
                Setup destination grid
                """
                bc_dst_left, bc_dst_right = get_bc(boundary_dst, boundary_dst_right, test_fun)
                
                dst_grid = get_grid(grid_dst_type, staggered_dst_grid, name="dst_u", boundaries=[bc_dst_left, bc_dst_right])
                
                
                if 0:
                    print(src_grid)
                    print(dst_grid)
                
                """
                Generate differential operator
                """
                u_diff = libpdefd.OperatorDiff1D(
                    diff_order = diff_order,        # Order of differentiation
                    min_approx_order = min_approx_order,    # Approximation order
                    src_grid=src_grid,      # Source variable to compute differential on
                    dst_grid=dst_grid,      # Destination variable to compute differential for
                ).bake()
                
                real_approx_order_equispaced = u_diff.real_approx_order_equispaced
                
                assert real_approx_order_equispaced >= min_approx_order
                
                if 0:
                    print("src_grid:")
                    print(src_grid)
                    print("dst_grid:")
                    print(dst_grid)
                    print("Operator (first 3 lines):")
                    print(u_diff.get_L()[0:3,:])
                
                
                    
                if plot_stencil != None:
                    import matplotlib.pyplot as plt
                    import libpdefd.plot_config as pc
                    
                    fig, ax = pc.setup(figsize=(10, 6))
                    
                    ax.axis('off')
                    
                    grid_line_width = 1.5
                    
                    def line(coord0, coord1, **kwargs):                        
                        x0 = coord0[0]
                        y0 = coord0[1]
                        x1 = coord1[0]
                        y1 = coord1[1]
                        
                        x0 = x0*0.8 + 0.1
                        x1 = x1*0.8 + 0.1
                        
                        y0 = y0*0.8 + 0.1
                        y1 = y1*0.8 + 0.1
                        
                        _ = plt.Line2D((x0, x1), (y0, y1), **kwargs)
                        plt.gca().add_line(_)
                        
                    
                    def arrow(coord0, coord1, color="black", width=0.002, **kwargs):                        
                        x0 = coord0[0]
                        y0 = coord0[1]
                        x1 = coord1[0]
                        y1 = coord1[1]
                        
                        x0 = x0*0.8 + 0.1
                        x1 = x1*0.8 + 0.1
                        
                        y0 = y0*0.8 + 0.1
                        y1 = y1*0.8 + 0.1
                        
                        plt.arrow(  x=x0, y=y0, dx=x1-x0, dy=y1-y0,
                                    width=width,
                                    length_includes_head = True,
                                    head_width=0.01, head_length=0.01,
                                    fc=color, ec=color,
                                )
                        
                    
                    def grid(x_stencil_dofs, x_dofs, y_grid):
                        line((0, y_grid), (1, y_grid), lw=grid_line_width, color="black")
                        
                        for x in x_stencil_dofs:
                            d = 0.01
                            color = "black" if x in x_dofs else "grey"
                            
                            line((x, y_grid-d), (x, y_grid+d), lw=grid_line_width, color=color)                        
                    
                    src_y = 1.0
                    src_x_stencil_dofs = (src_grid.x_stencil_dofs - domain_start) / domain_size
                    src_x_dofs = (src_grid.x_dofs - domain_start) / domain_size
                    grid(src_x_stencil_dofs, src_x_dofs, src_y)
                    
                    dst_y = 0.0
                    dst_x_stencil_dofs = (dst_grid.x_stencil_dofs - domain_start) / domain_size
                    dst_x_dofs = (dst_grid.x_dofs - domain_start) / domain_size
                    grid(dst_x_stencil_dofs, dst_x_dofs, dst_y)

                    c = u_diff.get_c()
                    for j in range(len(c)):
                        if c[j] != 0:
                            arrow((dst_x_dofs[j], 0.57), (dst_x_dofs[j], 0.03), width=0.002, color="green")
                    
                    color_id = 0
                    color_str_array = ["black", "gray"]
                    L = u_diff.get_L()
                    for j in range(L.shape[0]):
                        for i in range(L.shape[1]):
                            if L[j,i] != 0:
                                arrow((src_x_dofs[i], 0.97), (dst_x_dofs[j], 0.03), width=0.0015, color=color_str_array[color_id])
                                
                        color_id = (color_id + 1) % len(color_str_array)
                    
                    plt.tight_layout()
                    
                    
                    if plot_stencil == "show":
                        plt.show()
                        
                    elif plot_stencil == "file":
                        
                        id_str = "output_"
                        id_str += "src"
                        id_str += "_"+str(boundary_src)
                        id_str += "_"+str(grid_src_type)
                        id_str += "_"+("1" if staggered_src_grid else "0")
                        id_str += "__dst"
                        id_str += "_"+str(boundary_dst)
                        id_str += "_"+str(grid_dst_type)
                        id_str += "_"+("1" if staggered_dst_grid else "0")
                        id_str += "_"
                        id_str += "_difforder"+str(diff_order)
                        id_str += "_approxorder"+str(min_approx_order)
                        id_str += "_"
                        id_str += "_res"+str(res)
    
                        pc.savefig(id_str+".pdf", fig, verbose=True)
                
                
                """
                Increase frequency for higher order derivatives
                """
                u_0 = test_fun(src_grid.x_dofs, diff_order = 0)
                u_0 = libpdefd.VariableND(u_0)
                
                u_0_diff_num = u_diff.apply(u_0)
                u_0_diff = test_fun(dst_grid.x_dofs, diff_order = diff_order)
    
                if 0:
                    print(src_grid.x_dofs)
                    print(u_0)
                    sys.exit(1)
                
                if plot_solution:
                    import matplotlib.pyplot as plt
                    import libpdefd.plot_config as pc
                    
                    fig, ax = pc.setup(scale=1.5)
                    
                    ps = pc.PlotStyles()
                    
                    if 1:
                        plot_style = ps.getNextStyle()
                        plt.plot(src_grid.x_dofs, u_0, **plot_style, label="u0")
                    
                    if 1:
                        plot_style = ps.getNextStyle()
                        plt.plot(dst_grid.x_dofs, u_0_diff, **plot_style, label="u0_diff")
                    
                    if 1:
                        plot_style = ps.getNextStyle()
                        plt.plot(dst_grid.x_dofs, u_0_diff_num, **plot_style, label="u_0_diff_num")
                    
                    if 1:
                        plot_style = ps.getNextStyle()
                        plt.plot(dst_grid.x_dofs, u_0_diff - u_0_diff_num, **plot_style, label="error")
                    
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                
                if check_errors:
                    if error_norm == "max":
                        # Max norm
                        error = (u_0_diff_num - u_0_diff).reduce_maxabs()
                        
                    elif error_norm == "l2":
                        y = u_0_diff_num - u_0_diff
                        error = np.sqrt(np.sum(y*y)/dst_grid.num_dofs)
                        
                    else:
                        raise Exception("Norm '"+error_norm+"' not supported")
                    
                    # Interpolation points match support points?
                    if diff_order == 0:
                        if u_diff.all_dst_dof_points_aligned:
                            print("Aligned interpolation and support points detected. Error: "+str(error))
        
                            if error > 1e-12*domain_size:
                                raise Exception("Error too high for interpolation and aligned grids")
                            else:
                                print("Skipping convergence test (error is on order of numerical precision) since interpolation points match the support points!")
                                continue
        
                    conv = None
                    
                    if prev_error != None:
                        conv = prev_error / error 
                        
                    expected_conv = 2**real_approx_order_equispaced
        
                    print("res="+str(res)+"    error="+str(error)+"    conv="+str(conv)+"    expected_conv="+str(expected_conv)+"    expected_order="+str(real_approx_order_equispaced))
                    
                    prev_error = error
                    
                    if conv == None:
                        continue
                    
                    conv_list.append(conv)
            
            
            if check_errors:
                exact_matching_tested = False
                if diff_order == 0:
                    if u_diff.all_dst_dof_points_aligned:
                        exact_matching_tested = True
        
                if exact_matching_tested:
                    print("Skipping convergence test (error on order of numerical precision)")
        
                else:
                    print("Testing convergence: "+str(conv_list))
                    
                    if len(conv_list) == 0:
                        raise Exception("ERROR: No data to test for convergence")
                        
                    else:
                        
                        conv_threshold = 0.9
                        required_convergence_hits = 3
                        
                        if boundary_src in ["dirichlet", "neumann_extrapolated"] or boundary_src_right in ["dirichlet", "neumann_extrapolated"]:
                            conv_threshold -= 0.15
                        
                        if boundary_src == "neumann_extrapolated" or boundary_src_right == "neumann_extrapolated":
                            if neumann_diff_order > 1:
                                conv_threshold -= 0.2
                            
                        if diff_order + real_approx_order_equispaced >= 6:
                            conv_threshold -= 0.1
                            required_convergence_hits -= 1
                        
                        if expected_conv >= 32:
                            conv_threshold -= 0.1
                            required_convergence_hits -= 1
                        
                        # Require at least one hit
                        required_convergence_hits = max(1, required_convergence_hits)
                        
                        # Ensure minimum convergence threshold
                        conv_threshold = max(0.6, conv_threshold)
                        
                        conv_min = conv_threshold*expected_conv
                        conv_max = (2.0-conv_threshold)*expected_conv
                        
                        if boundary_src in ["symmetric"] or boundary_src_right in ["symmetric"]:
#                            if min_approx_order >= 3:
                            conv_max *= 2.0
                        
                        if boundary_src in ["neumann_extrapolated"] or boundary_src_right in ["neumann_extrapolated"]:
#                            if min_approx_order >= 3:
                            conv_max *= 4.0
                        
                        elif boundary_src in ["dirichlet", "neumann_extrapolated"] or boundary_src_right in ["dirichlet", "neumann_extrapolated"]:
                            if boundary_dst in ["symmetric", "periodic"] or boundary_dst_right in ["symmetric", "periodic"]:
                                """
                                We also relax the maximum convergence threshold for other special cases
                                """
                                conv_max *= 2.0
                                
                                """
                                Special case for diff order 0 and perfectly aligned grids
                                """
                                if staggered_src_grid == staggered_dst_grid:
                                    if diff_order == 0:
                                        conv_max *= 2.0
                                    
                    
                        if grid_dst_type == "homemade_nonequidistant":
                            """
                            Working with nonequidistant destination grids can be a problem for convergence.
                            Hence, we relax things more for this
                            """
                            conv_max *= 2.0
                            conv_min *= 0.5
                            if boundary_src in ["periodic"]:
                                print("Relaxing convergence requirements due to nonequidistant nodes")
                        
                        
                        print("conv_min: "+str(conv_min))
                        print("conv_max: "+str(conv_max))
                        
                        hit_counter = 0
                        
                        for conv in conv_list:
                            if conv > conv_min and conv < conv_max:
                                hit_counter += 1
                        
                        
                        print(" + conv_threshold: "+str(conv_threshold))
                        print(" + number of OKish convergence: "+str(hit_counter))
        
                        if diff_order == 0:
                            if u_diff.all_dst_dof_points_aligned:
                                print("Skipping error tests, since error has been tested to be 0")
                                continue
        
                        if hit_counter < required_convergence_hits:
                            raise Exception("Less than 2x approximation order reached")
                    
