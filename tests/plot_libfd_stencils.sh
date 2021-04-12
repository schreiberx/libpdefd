#! /bin/bash

python test_libfd.py --boundary-src=dirichlet,neumann_extrapolated,symmetric --boundary-dst=dirichlet,neumann_extrapolated,symmetric --grid-src-type=auto --grid-dst-type=auto --staggered-src-grid=True,False --staggered-dst-grid=True,False --diff-order=1,2 --min-approx-order=1,2 --start-res2=3 --end-res2=4 --check-error=False --plot-stencil=file

