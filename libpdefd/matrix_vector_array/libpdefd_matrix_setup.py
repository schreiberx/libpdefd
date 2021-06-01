
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import sys
import time

"""
Use this hack to use these python files also without libpdefd
"""
try:
    import libpdefd.matrix_vector_array.libpdefd_vector_array as libpdefd_vector_array
except:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    import libpdefd_vector_array as libpdefd_vector_array
    sys.path.pop()



"""
This module implements the class for sparse matrices during setup.

Since the main computational workload is in matrix_compute, typically
no significant optimizations are required in this matrix.
"""

class matrix_sparse:
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
            format_override = False
    ):
        if _data != None:
            """
            _data refers to an already existing matrix which can be in different formats
            
            Everything will be converted to a lil_matrix 
            """
            if isinstance(_data, sparse.lil_matrix):
                self._matrix_lil = _data
                
            elif isinstance(_data, float):
                self._matrix_lil = sparse.lil_matrix([[_data]])
            
            #elif isinstance(_data, np.ndarray):
            #    self._matrix_lil = sparse.lil_matrix(_data)
            #    raise Exception("Shouldn't be required")
            
            elif format_override:
                self._matrix_lil = _data
                
            else:
                raise Exception("Unsupported type '"+str(type(_data))+"' to initialize data")
            
            self.shape = self._matrix_lil.shape
            if not format_override:
                assert isinstance(self._matrix_lil, sparse.lil_matrix)
            return
        
        self.shape = shape
        
        if self.shape == None:
            return
        
        self._matrix_lil = sparse.lil_matrix(self.shape, dtype=dtype)
        assert isinstance(self._matrix_lil, sparse.lil_matrix)
    
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            retval = self._matrix_lil[key]
            if isinstance(retval, float):
                return retval

            return matrix_sparse(_data = retval)
            #print(type(self._matrix_lil[key]))
            #return self._matrix_lil[key]
            #return matrix_sparse(_data = self._matrix_lil[key])
        
        """
        if isinstance(key, tuple):
            m = matrix_sparse()
            m._matrix_lil = self._matrix_lil[key]
            m.shape = m._matrix_lil.shape
            return m
        """
    
        return self._matrix_lil[key]
    
    
    def __setitem__(self, key, data):
        
        if isinstance(data, matrix_sparse):
            # Note that we need to convert scipy sparse arrays to numpy arrays before assigning them
            self._matrix_lil[key] = data.to_numpy_array()
            return self
        
        if isinstance(data, np.ndarray) or isinstance(data, int) or isinstance(data, float):
            self._matrix_lil[key] = data
            return self
        
        raise Exception("Shouldn't happen")
    
    
    def getrow_asarray(self, idx):
        return self._matrix_lil[idx,:]
    
    
    def getcol_asarray(self, idx):
        return self._matrix_lil[:,idx].toarray()[:,0]
    
    
    def __add__(self, data):
        assert isinstance(self._matrix_lil, sparse.lil_matrix)
        
        mdata = self._matrix_lil + data._matrix_lil
        return matrix_sparse(_data = mdata)
    
    
    def __iadd__(self, data):
        if isinstance(data, matrix_sparse):
            self._matrix_lil -= data._matrix_lil
            """
            Important: Do not do any conversions to lil here because of performance reasons
            This would slow this down extremely
            """
            return self
        
        raise Exception("Shouldn't happen")
    
    
    def __sub__(self, data):
        assert isinstance(self._matrix_lil, sparse.lil_matrix)

        mdata = self._matrix_lil - data._matrix_lil
        return matrix_sparse(_data = mdata)
    
    
    def __isub__(self, data):
        if isinstance(data, matrix_sparse):
            self._matrix_lil -= data._matrix_lil
            return self
        
        raise Exception("Shouldn't happen")
    
    
    def __truediv__(self, data):
        if isinstance(data, float) or isinstance(data, int):
            return matrix_sparse(_data = self._matrix_lil/data)
        
        raise Exception("Shouldn't happen")
    
    
    def __mul__(self, data):
        if isinstance(data, float) or isinstance(data, int):
            return matrix_sparse(_data = self._matrix_lil*data)

        raise Exception("Shouldn't happen")
    
    
    def reduce_min(self):
        return np.min(self._matrix_lil)
    
    def reduce_max(self):
        return np.max(self._matrix_lil)
   
   
    def kron(self, data):
        if isinstance(data, matrix_sparse):
            if 1:
                m = sparse.kron(self._matrix_lil, data._matrix_lil)
                
                """
                Important: Do not do any conversions to lil here because of performance reasons
                This would slow this down extremely
                """
                return matrix_sparse(_data = m, format_override=True)
            
            else:
                a = sparse.csr_matrix(self._matrix_lil)
                b = sparse.csr_matrix(data._matrix_lil)
                k = sparse.kron(a, b)
                m = sparse.lil_matrix(k)
                
            return matrix_sparse(_data = m)
        
        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    
    def dot(self, data):
        """
        Compute
            M*x
        by reshaping 'x' so that it fits the number of rows in 'M'
        """
        
        if isinstance(data, libpdefd_vector_array._vector_array_base):
            """
            If input is an ndarray, return also an libpdefd_vector_array
            """
            d = self._matrix_lil.dot(data._data)
            assert isinstance(d, np.ndarray)
            return libpdefd_vector_array.vector_array(d)
        
        if isinstance(data, np.ndarray) and False:
            """
            If input is an ndarray, return also an ndarray
            """
            d = self._matrix_lil.dot(data)
            assert isinstance(d, np.ndarray)
            return d

        if isinstance(data, matrix_sparse):
            d = self._matrix_lil.dot(data._matrix_lil)
            #d = sparse.lil_matrix(d)
            """
            Important: Do not do any conversions to lil here because of performance reasons
            This would slow this down extremely
            """
            return matrix_sparse(_data = d, format_override=True)
        
        raise Exception("Unsupported type data '"+str(type(data))+"'")


    def __str__(self):
        retstr = ""
        retstr += "PYSMmatrixsetup: "
        retstr += " shape: "+str(self.shape)
        #retstr += "\n"
        #retstr += str(self._matrix_lil)
        return retstr

    def to_numpy_array(self):
        return self._matrix_lil.toarray()



def eye_sparse(shape):
    """
    Performance bug: creating LIL eye matrices is extremely slow!
    Therefore, we switch to CSR here.
    """
    return matrix_sparse(_data=sparse.eye(shape, format='csr'), format_override=True)


