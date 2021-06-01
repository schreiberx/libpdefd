#! /usr/bin/env python3

import sys, os
import time

"""
Use this hack to use these python files also without libpdefd
"""
try:
    import libpdefd.array_matrix.libpdefd_vector_array as libpdefd_vector_array
    import libpdefd.array_matrix.libpdefd_matrix_setup as libpdefd_matrix_setup
    import libpdefd.array_matrix.libpdefd_matrix_compute as libpdefd_matrix_compute
    
except:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    import libpdefd_vector_array as libpdefd_vector_array
    import libpdefd_matrix_setup as libpdefd_matrix_setup
    import libpdefd_matrix_compute as libpdefd_matrix_compute
    sys.path.pop()



"""
Problem size
"""
N = 2

if len(sys.argv) > 1:
    N = int(sys.argv[1])

"""
Number of iterations
"""
K = 1

if len(sys.argv) > 2:
    K = int(sys.argv[2])


print("")
print("*"*80)
print("Array A")
print("*"*80)
a = libpdefd_vector_array.array_zeros((N,N+1,N+2))
al = a.num_elements()

for z in range(a.shape[0]):
    for y in range(a.shape[1]):
        for x in range(a.shape[2]):
            a[z,y,x] = x + y*a.shape[1] + z*(a.shape[1]*a.shape[0])

print(a)



print("")
print("*"*80)
print("Array B")
print("*"*80)
b = libpdefd_vector_array.array_zeros((N+2,N+3,N+4))
bl = b.num_elements()

for z in range(b.shape[0]):
    for y in range(b.shape[1]):
        for x in range(b.shape[2]):
            b[z,y,x] = x + y*b.shape[1] + z*(b.shape[1]*b.shape[0])
print(b)



print("")
print("*"*80)
print("Array C")
print("*"*80)
c = b*2.0
c = c.flatten()
print(c)



print("")
print("*"*80)
print("Matrix Sparse")
print("*"*80)

print(" + allocation")
m_setup = libpdefd_matrix_setup.matrix_sparse(shape=(bl, al))

print(" + setup")
for i in range(min(al, bl)):
    m_setup[i,i] = -2

for i in range(min(al, bl)-1):
    m_setup[i,i+1] = 1
    m_setup[i,i-1] = 3

print(m_setup)


print("*"*80)
print("Matrix Compute")
print("*"*80)

m_compute = libpdefd_matrix_compute.matrix_sparse(m_setup)
print(m_compute)



print("*"*80)
print("Benchmarks")
print("*"*80)

time_start = time.time()

for k in range(K):
    print("Iteration: "+str(k))
    
    if 1:
        print("MUL test 1")
        retval = m_compute.dot__DEPRECATED(a.flatten())
        assert retval.shape == c.shape
    
    
    if 1:
        print("MUL test 2")
        retval = m_compute.dot_add_reshape(a, c, b.shape)
        assert retval.shape == b.shape
    
    
    if 1:
        print("MUL test 3")
        retval = m_compute.dot_add_reshape(a, c, b.shape)
        assert retval.shape == b.shape


time_end = time.time()

print("Seconds: "+str(time_end-time_start))

print("*"*80)
print("FIN")
print("*"*80)