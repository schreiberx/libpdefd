
import numpy as np
import scipy as sp
import scipy.sparse as sparse

"""
Use this hack to use these python files also without libpdefd
"""
try:
    import libpdefd.array_matrix.libpdefd_array as libpdefd_array
except:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    import libpdefd_array as libpdefd_array
    sys.path.pop()


"""
This module implements two different classes:

 * matrices used during setups
 
 * matrices used for applying operators
 
The main difference to 'array' is, that these are sparse matrices
"""

class matrix_sparse_for_setup:
    """
    This matrix uses a layout suited to setup the matrix itself.
    """
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(
            self,
            shape = None,
            dtype = None
    ):
        self.shape = shape
        
        if self.shape == None:
            return
        
        self._matrix_lil = sparse.lil_matrix(self.shape, dtype = dtype)
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._matrix_lil[key]
        
        if isinstance(key, tuple):
            m = matrix_sparse_for_setup()
            m.matrix_lil = self._matrix_lil[key]
            return m
            
        return self._matrix_lil[key]
    
    def __setitem__(self, key, data):
        if isinstance(data, matrix_sparse_for_setup):
            self._matrix_lil[key] = data._matrix_lil
            return self
        
        self._matrix_lil[key] = data
        return self
    
    def __add__(self, data, datab):
        m = matrix_sparse_for_setup()
        m.matrix_lil = data + datab
        return m
    
    def __sub__(self, data, datab):
        m = matrix_sparse_for_setup()
        m.matrix_lil = data - datab
        return m
    
    def __iadd__(self, data):
        self._matrix_lil += data
        return self
    
    def __isub__(self, data):
        self._matrix_lil -= data
        return self
    
    def __str__(self):
        retstr = "shape: "+str(self.shape)
        #retstr += "\n"
        #retstr += str(self._matrix_lil)
        return retstr

    def to_numpy_array(self):
        return self._matrix_lil.toarray()



class matrix_sparse_for_compute:
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
        
        if isinstance(data, matrix_sparse_for_setup):
            self._matrix_csr = sparse.csr_matrix(data._matrix_lil)
            self.shape = self._matrix_csr.shape
            return
        
        if isinstance(data, matrix_sparse_for_compute):
            self._matrix_csr = data._matrix_csr
            self.shape = self._matrix_csr.shape
            return

        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    
    def __iadd__(self, data):
        if isinstance(data, matrix_sparse_for_compute):
            self._matrix_csr += data._matrix_csr
            return self

        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    def __isub__(self, data):
        if isinstance(data, matrix_sparse_for_compute):
            self._matrix_csr -= data._matrix_csr
            return self

        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    def __str__(self):
        retstr = "shape: "+str(self.shape)
        #retstr += "\n"
        #retstr += str(self._matrix_lil)
        return retstr
    
    
    def reduce_min(self):
        return np.min(self._matrix_csr)
    
    def reduce_max(self):
        return np.max(self._matrix_csr)
    
    
    def kron(self, data):
        if isinstance(data, matrix_sparse_for_compute):
            k = sparse.kron(self._matrix_csr, data._matrix_csr)
            k = sparse.csr_matrix(k)
            return matrix_sparse_for_compute(k)
        
        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    
    def dot(self, data):
        """
        Compute
            M*x
        by reshaping 'x' so that it fits the number of rows in 'M'
        """
        
        if isinstance(data, libpdefd_array._array_base):
            """
            If input is an ndarray, return also an libpdefd_array
            """
            d = self._matrix_csr.dot(data._data)
            assert isinstance(d, np.ndarray)
            return libpdefd_array.array(d)
        
        if isinstance(data, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """
            d = self._matrix_csr.dot(data)
            assert isinstance(d, np.ndarray)
            return d

        if isinstance(data, matrix_sparse_for_compute):
            d = self._matrix_csr.dot(data._matrix_csr)
            return matrix_sparse_for_compute(d)
        
        raise Exception("Unsupported type data '"+str(type(data))+"'")

    
    def dot_add(self, x, c):
        """
        Compute
            M*x + c
        """
        
        if type(x) != type(c):
            raise Exception("x and c must have same type")
        
        if isinstance(x, libpdefd_array._array_base):
            d = self._matrix_csr.dot(x._data) + c._data
            return libpdefd_array.array(d)
        
        if isinstance(x, np.ndarray) and False:
            d = self._matrix_csr.dot(x) + c
            assert isinstance(d, np.ndarray)
            return d
        
        if isinstance(x, matrix_sparse_for_compute):
            d = self._matrix_csr.dot(x._matrix_csr) + c._data
            return matrix_sparse_for_compute(d)
        
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
        
        if isinstance(x, libpdefd_array._array_base):
            d = (self._matrix_csr.dot(x._data.flatten()) + c._data).reshape(dst_shape)
            
            assert isinstance(d, np.ndarray)
            return libpdefd_array.array(d)
        
        if isinstance(x, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """
            d = (self._matrix_csr.dot(x.flatten()) + c).reshape(dst_shape)
            
            assert isinstance(d, np.ndarray)
            return d
        
        if isinstance(x, matrix_sparse_for_compute):
            raise Exception("This case shouldn't exist")
        
        raise Exception("Unsupported type data '"+str(type(x))+"' for x")

    def to_numpy_array(self):
        return self._matrix_csr.toarray()



def eye_compute(shape):
    return matrix_sparse_for_compute(sparse.csr_matrix(sparse.eye(shape)))

