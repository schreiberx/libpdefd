import numpy as np


"""
The vector-array class is one which
 * stores multi-dimensional array data representing spatial information and
 * allows efficient matrix-vector multiplications with the compute matrix class
"""

class _vector_array_base:
    """
    Array container to store varying data during simulation
    
    This data is stored on a regular high(er) dimensional Cartesian grid.
    """
    
    def __init__(self, data = None):
        self._data = data
        
        if self._data is not None:
            if isinstance(data, self.__class__):
                self._data = data._data
            else:
                self._data = data

        self.shape = None

        if self._data is not None:
            self.shape = self._data.shape
    
    def __getitem__(self, i):
        return self._data[i]
    
    def __setitem__(self, i, data):
        self._data[i] = data
    
    
    def __add__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(self._data + a._data)
        
        if isinstance(a, float):
            return self.__class__(self._data + a)
        
        raise Exception("Unsupported type '"+str(type(a))+"'")
    
    def __iadd__(self, a):
        if isinstance(a, self.__class__):
            self._data += a._data
        else:
            self._data += a
        
        return self
    
    def __radd__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(a._data + self._data)
        return self.__class__(a + self._data)
    
    
    def __sub__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(self._data - a._data)
        return self.__class__(self._data - a)
        
    
    def __isub__(self, a):
        if isinstance(a, self.__class__):
            self._data -= a._data
        else:
            self._data -= a
            
        return self
    
    def __rsub__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(a._data - self._data)
        return self.__class__(a - self._data)
    
    
    def __mul__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(a._data * self._data)
        return self.__class__(a * self._data)
    
    def __imul__(self, a):
        if isinstance(a, self.__class__):
            self._data *= a._data
        else:
            self._data *= a
        
        return self
    
    def __rmul__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(a._data * self._data)
        return self.__class__(a * self._data)
    
    
    def __pow__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(self._data ** a._data)
        return self.__class__(self._data ** a)
    
    def __truediv__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(self._data / a._data)
        return self.__class__(self._data / a)
    
    def __rtruediv__(self, a):
        if isinstance(a, self.__class__):
            return self.__class__(a._data / self._data)
        return self.__class__(a / self._data)
    
    def __itruediv__(self, a):
        if isinstance(a, self.__class__):
            self._data += a._data
            return self
        self._data += a
        return self
    
    def __neg__(self):
        return self.__class__(-self._data)
    
    def __pos__(self):
        return self.__class__(self._data.copy())
    
    def __str__(self):
        retstr = "PYSMarray: "
        retstr += str(self.shape)
        return retstr
    
    def abs(self):
        return array(np.abs(self._data))
    
    def reduce_min(self):
        return np.min(self._data)
    
    def reduce_minabs(self):
        return np.min(np.abs(self._data))
    
    def reduce_max(self):
        return np.max(self._data)
    
    def reduce_maxabs(self):
        return np.max(np.abs(self._data))
    
    
    def copy(self):
        return self.__class__(self._data.copy())
    
    def kron(self, data):
        assert isinstance(data, _vector_array_base)
        d = np.kron(self._data, data._data)
        return self.__class__(d)
    
    def set_all(self, scalar_value):
        self._data[:] = scalar_value
    
    def to_numpy_array(self):
        """
        Return numpy array
        """
        assert isinstance(self._data, np.ndarray)
        return self._data

    def flatten(self):
        return self.__class__(self._data.flatten())
    
    def num_elements(self):
        return np.prod(self.shape)



def vector_array(param, dtype=None, *args, **kwargs):
    retval = _vector_array_base()
        
    if isinstance(param, np.ndarray):
        retval._data = np.copy(param)
        retval.shape = param.shape
        return retval
    
    if isinstance(param, _vector_array_base):
        retval._data = np.copy(param._data)
        retval.shape = param.shape
        return retval

    if isinstance(param, list):
        retval._data = np.array(param, dtype=dtype)
        retval.shape = retval._data.shape
        return retval
    
    raise Exception("Type '"+str(type(param))+"' of param not supported")


def vector_array_zeros(shape, dtype=None):
    """
    Return array of shape with zeros
    """
    retval = _vector_array_base()
    retval._data = np.zeros(shape, dtype=dtype)
    retval.shape = retval._data.shape
    return retval


def vector_array_zeros_like(data, dtype=None):
    """
    Return zero array of same shape as data
    """
    return vector_array_zeros(data.shape, dtype=dtype)


def vector_array_ones(shape, dtype=None):
    """
    Return array of shape with ones
    """
    retval = _vector_array_base()
    retval._data = np.ones(shape, dtype=dtype)
    retval.shape = retval._data.shape
    return retval

