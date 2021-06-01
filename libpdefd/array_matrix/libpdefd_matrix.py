
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import sys

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
            dtype = None,
            _data = None,
    ):
        if _data != None:
            if isinstance(_data, sparse.lil_matrix):
                self._matrix_lil = _data
                
            elif isinstance(_data, float):
                self._matrix_lil = sparse.lil_matrix([[_data]])
            
            elif isinstance(_data, np.ndarray):
                self._matrix_lil = sparse.lil_matrix(_data)
                raise Exception("Shouldn't be required")
                
            else:
                raise Exception("Unsupported type '"+str(type(_data))+"' to initialize data")
            
            self.shape = self._matrix_lil.shape
            return
        
        self.shape = shape
        
        if self.shape == None:
            return
        
        self._matrix_lil = sparse.lil_matrix(self.shape, dtype=dtype)
    
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            retval = self._matrix_lil[key]
            if isinstance(retval, float):
                return retval
            
            return matrix_sparse_for_setup(_data = retval)
            #print(type(self._matrix_lil[key]))
            #return self._matrix_lil[key]
            #return matrix_sparse_for_setup(_data = self._matrix_lil[key])
        
        """
        if isinstance(key, tuple):
            m = matrix_sparse_for_setup()
            m._matrix_lil = self._matrix_lil[key]
            m.shape = m._matrix_lil.shape
            return m
        """
    
        return self._matrix_lil[key]
    
    
    def __setitem__(self, key, data):
        
        if isinstance(data, matrix_sparse_for_setup):
            # Note that we need to convert scipy sparse arrays to numpy arrays before assigning them
            self._matrix_lil[key] = data.to_numpy_array()
            return self
        
        if isinstance(data, np.ndarray) or isinstance(data, int) or isinstance(data, float):
            self._matrix_lil[key] = data
            return self
        
        raise Exception("TODO")
        return self
    
    def __add__(self, data):
        return matrix_sparse_for_setup(_data = self._matrix_lil + data)
    
    def __sub__(self, data, datab):
        return matrix_sparse_for_setup(_data = data - datab)
    
    def __iadd__(self, data):
        if isinstance(data, float) or isinstance(data, int):
            mdata = sparse.lil_matrix(self._matrix_lil.toarray() + data)
            return matrix_sparse_for_setup(_data = mdata)
        
        if isinstance(data, matrix_sparse_for_setup):
            d = sparse.lil_matrix(self._matrix_lil + data._matrix_lil)
            return matrix_sparse_for_setup(_data = d)
        
        raise Exception("TODO")
        return matrix_sparse_for_setup(_data = mdata)
    
    def __isub__(self, data):
        if isinstance(data, float) or isinstance(data, int):
            mdata = sparse.csr_matrix(self._matrix_lil.toarray() - data)
            return matrix_sparse_for_setup(_data = mdata)
        
        if isinstance(data, matrix_sparse_for_setup):
            d = sparse.lil_matrix(self._matrix_lil - data._matrix_lil)
            return matrix_sparse_for_setup(_data = d)
        
        if isinstance(data, np.ndarray):
            mdata = sparse.lil_matrix(self._matrix_lil - data)
            return matrix_sparse_for_setup(_data = mdata)
        
        return matrix_sparse_for_setup(_data = self._matrix_lil - data)
    
    def __truediv__(self, data):
        if isinstance(data, float) or isinstance(data, int):
            return matrix_sparse_for_setup(_data = self._matrix_lil/data)
        
        raise Exception("TODO")
        return matrix_sparse_for_setup(_data = self._matrix_lil/data)
    
    def __mul__(self, data):
        if isinstance(data, float) or isinstance(data, int):
            return matrix_sparse_for_setup(_data = self._matrix_lil*data)
        
        raise Exception("TODO")
        return matrix_sparse_for_setup(_data = self._matrix_lil*data)
    
    def reduce_min(self):
        return np.min(self._matrix_lil)
    
    def reduce_max(self):
        return np.max(self._matrix_lil)
   
   
    def kron(self, data):
        if isinstance(data, matrix_sparse_for_setup):
            m = sparse.lil_matrix(sparse.kron(self._matrix_lil, data._matrix_lil))
            return matrix_sparse_for_setup(_data = m)
        
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
            d = self._matrix_lil.dot(data._data)
            assert isinstance(d, np.ndarray)
            return libpdefd_array.array(d)
        
        if isinstance(data, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """
            d = self._matrix_lil.dot(data)
            assert isinstance(d, np.ndarray)
            return d

        if isinstance(data, matrix_sparse_for_setup):
            d = self._matrix_lil.dot(data._matrix_lil)
            d = sparse.lil_matrix(d)
            return matrix_sparse_for_setup(_data = d)
        
        raise Exception("Unsupported type data '"+str(type(data))+"'")


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
    
    
    def dot__DEPRECATED(self, data):
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

    
    def dot_add__DEPRECATED(self, x, c):
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



def eye_setup(shape):
    return matrix_sparse_for_setup(_data=sparse.eye(shape, format='lil'))

