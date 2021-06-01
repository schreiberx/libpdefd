
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import sys

"""
Use this hack to use these python files also without libpdefd
"""
try:
    import libpdefd.matrix_vector_array.libpdefd_vector_array as libpdefd_vector_array
    import libpdefd.matrix_vector_array.libpdefd_matrix_setup as libpdefd_matrix_setup
except:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    import libpdefd_vector_array as libpdefd_vector_array
    import libpdefd_matrix_setup as libpdefd_matrix_setup
    sys.path.pop()



"""
This module implements the compute class for sparse matrices

Hence, this should be highly optimized
"""


class matrix_sparse:
    """
    This class is using a format which is suited for fast matrix-vector multiplications.
    """
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(self, data):
        
        if isinstance(data, np.ndarray):
            self._matrix_csr = sparse.csr_matrix(data)
            self.shape = self._matrix_csr.shape
            return
                
        if isinstance(data, sparse.csr_matrix):
            self._matrix_csr = data
            self.shape = self._matrix_csr.shape
            return
        
        if isinstance(data, sparse.lil_matrix):
            self._matrix_csr = sparse.csr_matrix(data)
            self.shape = self._matrix_csr.shape
            return
        
        if isinstance(data, libpdefd_matrix_setup.matrix_sparse):
            self._matrix_csr = sparse.csr_matrix(data._matrix_lil)
            self.shape = self._matrix_csr.shape
            return
        
        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    
    def __str__(self):
        retstr = ""
        retstr += "PYSMmatrixcompute: "
        retstr += " shape: "+str(self.shape)
        #retstr += "\n"
        #retstr += str(self._matrix_lil)
        return retstr
    
    
    def dot__DEPRECATED(self, data):
        """
        Compute
            M*x
        by reshaping 'x' so that it fits the number of rows in 'M'
        """
        
        if isinstance(data, libpdefd_vector_array._vector_array_base):
            """
            If input is an ndarray, return also an libpdefd_vector_array
            """
            d = self._matrix_csr.dot(data._data)
            assert isinstance(d, np.ndarray)
            return libpdefd_vector_array.vector_array(d)
        
        if isinstance(data, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """
            d = self._matrix_csr.dot(data)
            assert isinstance(d, np.ndarray)
            return d

        if isinstance(data, matrix_sparse):
            d = self._matrix_csr.dot(data._matrix_csr)
            return matrix_sparse(d)
        
        raise Exception("Unsupported type data '"+str(type(data))+"'")

    
    def dot_add__DEPRECATED(self, x, c):
        """
        Compute
            M*x + c
        """
        
        if type(x) != type(c):
            raise Exception("x and c must have same type")
        
        if isinstance(x, libpdefd_vector_array._vector_array_base):
            d = self._matrix_csr.dot(x._data) + c._data
            return libpdefd_vector_array.vector_array(d)
        
        if isinstance(x, np.ndarray) and False:
            d = self._matrix_csr.dot(x) + c
            assert isinstance(d, np.ndarray)
            return d
        
        if isinstance(x, matrix_sparse):
            d = self._matrix_csr.dot(x._matrix_csr) + c._data
            return matrix_sparse(d)
        
        raise Exception("Unsupported type data '"+str(type(x))+"'")
    
    
    def dot_add_reshape(self, x, c, dst_shape):
        """
        Compute
            M*x + c
        by reshaping 'x' so that it fits the number of rows in 'M'.
        
        Reshape the output to fit dst_shape.
        """
        
        if type(x) != type(c):
            raise Exception("x and c must have same type")
        
        if isinstance(x, libpdefd_vector_array._vector_array_base):
            d = (self._matrix_csr.dot(x._data.flatten()) + c._data).reshape(dst_shape)
            
            assert isinstance(d, np.ndarray)
            return libpdefd_vector_array.vector_array(d)
        
        if isinstance(x, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """
            d = (self._matrix_csr.dot(x.flatten()) + c).reshape(dst_shape)
            
            assert isinstance(d, np.ndarray)
            return d
        
        if isinstance(x, matrix_sparse):
            raise Exception("This case shouldn't exist")
        
        raise Exception("Unsupported type data '"+str(type(x))+"' for x")

    def to_numpy_array(self):
        return self._matrix_csr.toarray()

