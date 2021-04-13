import numpy as np
import scipy as sp
import scipy.sparse as sparse




class MatrixSparseSetup:
    """
    This matrix should be used during setup.
    It's not necessary to optimize the access to this matrix
    """
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(
            self,
            shape = None,
            dtype = None
    ):
        if shape == None:
            return
        
        self._matrix_lil = sparse.lil_matrix(shape, dtype = dtype)
    

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._matrix_lil[key]
        
        if isinstance(key, tuple):
            m = MatrixSparseSetup()
            m.matrix_lil = self._matrix_lil[key]
            return m
            
        return self._matrix_lil[key]
    
    def __setitem__(self, key, data):
        if isinstance(data, MatrixSparseSetup):
            self._matrix_lil[key] = data._matrix_lil
            return self
            
        self._matrix_lil[key] = data
        return self
    
    def __str__(self):
        return str(self._matrix_lil)
    
    
    def __add__(self, data, datab):
        m = MatrixSparseSetup()
        m.matrix_lil = data + datab
        return m
    
    def __sub__(self, data, datab):
        m = MatrixSparseSetup()
        m.matrix_lil = data - datab
        return m
    
    
    def __iadd__(self, data):
        self._matrix_lil += data
        return self
    
    def __isub__(self, data):
        self._matrix_lil -= data
        return self
    
    
    def min(self):
        return np.min(self._matrix_lil)
    
    def max(self):
        return np.max(self._matrix_lil)
    
    

class MatrixSparseCompute:
    """
    Return a matrix format which is suited for the computational part
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
        
        if isinstance(data, MatrixSparseSetup):
            self._matrix_csr = sparse.csr_matrix(data._matrix_lil)
            self.shape = self._matrix_csr.shape
            return
        
        if isinstance(data, MatrixSparseCompute):
            self._matrix_csr = data._matrix_csr
            self.shape = self._matrix_csr.shape
            return

        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    
    def min(self):
        return np.min(self._matrix_csr)
    
    def max(self):
        return np.max(self._matrix_csr)
    
    
    def __iadd__(self, data):
        if isinstance(data, MatrixSparseCompute):
            self._matrix_csr += data._matrix_csr
            return self

        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    def __isub__(self, data):
        if isinstance(data, MatrixSparseCompute):
            self._matrix_csr -= data._matrix_csr
            return self

        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    
    def kron(self, data):
        if isinstance(data, MatrixSparseCompute):
            k = sparse.kron(self._matrix_csr, data._matrix_csr)
            k = sparse.csr_matrix(k)
            return MatrixSparseCompute(k)
    
        raise Exception("Unsupported type data '"+str(type(data))+"'")
    
    
    def dot(self, data):
        
        if isinstance(data, np.ndarray):
            """
            If input is an ndarray, return also an ndarray
            """
            d = self._matrix_csr.dot(data)
            assert isinstance(d, np.ndarray)
            return d

        if isinstance(data, MatrixSparseCompute):
            d = self._matrix_csr.dot(data._matrix_csr)
            return MatrixSparseCompute(d)
        
        raise Exception("Unsupported type data '"+str(type(data))+"'")


def to_sparse_matrix_for_compute(data, *args, **kwargs):
    """
    Convert
    """
    
    return MatrixSparseCompute(data, *args, **kwargs)

    if isinstance(data, MatrixSparseSetup):
        return sparse.csr_matrix(data._matrix_lil, *args, **kwargs)

    return sparse.csr_matrix(data, *args, **kwargs)


def eye(shape):
    return MatrixSparseCompute(sparse.csr_matrix(sparse.eye(shape)))



if __name__ == "__main__":
    
    m = MatrixSparseSetup((2, 2))
    m[1,1] = 123
    assert m[1,1] == 123

    m[1,1] += 1.
    assert m[1,1] == 124
    
    print("Tests passed")

