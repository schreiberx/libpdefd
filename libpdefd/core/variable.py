import numpy as np
from libpdefd.core.gridinfo import *
import libpdefd.array_matrix.libpdefd_array as libpdefd_array


class _VariableND_Base:
    """
    Provide basic arithmetic operations for variable
    """
    def __init__(self, data = None, shape = None, name = None):
        self._name = name
        self._data = data
        self.shape = shape
        
        if self.shape is None:
            if self._data is not None:
                self.shape = self._data.shape
            
    
    def reciprocal(self):
        data = 1.0/self._data
        return self.__class__(data=data)

    def copy(self):
        return self.__class__(name=self._name, data=self._data.copy())
        
        
    def __getitem__(self, i):
        return self._data[i]


    def __setitem__(self, i, data):
        self._data[i] = data


    def __add__(self, a):
        if isinstance(a, self.__class__):
            data = self._data + a._data
        else:
            data = self._data + a
        return self.__class__(data=data)

    def __sub__(self, a):
        if isinstance(a, self.__class__):
            data = self._data - a._data
        else:
            data = self._data - a
        return self.__class__(data=data)

    def __mul__(self, a):
        if isinstance(a, self.__class__):
            data = self._data * a._data
 
        else:
            data = self._data * a
            
        return self.__class__(data=data)


    def __rmul__(self, a):
        """
        Support
            scalar * VariableND()
        operators
        """
        if isinstance(a, self.__class__):
            data = a._data * self._data
 
        else:
            data = a * self._data
            
        return self.__class__(data=data)


    def __truediv__(self, a):
        """
        Support regular division (not floored/integer division)
        """
        if isinstance(a, self.__class__):
            data = self._data / a._data
        else:
            data = self._data / a

        return self.__class__(data=data)


    def __rtruediv__(self, a):
        """
        Support regular division (not floored/integer division)
        """
        if isinstance(a, self.__class__):
            data = a._data / self._data
        else:
            data = a / self._data

        return self.__class__(data=data)

    def __iadd__(self, a):
        """
        Support
            +=
        operators
        """
        if isinstance(a, self.__class__):
            self._data += a._data
        else:
            self._data += a

        return self

    def __isub__(self, a):
        if isinstance(a, self.__class__):
            self._data -= a._data
        else:
            self._data -= a
        return self

    def __imul__(self, a):
        if isinstance(a, self.__class__):
            self._data *= a._data
        else:
            self._data *= a
        return self

    def __idiv__(self, a):
        if isinstance(a, self.__class__):
            self._data /= a._data
        else:
            self._data /= a
        return self

    def __pow__(self, a):
        if isinstance(a, self.__class__):
            data = self._data ** a._data
 
        else:
            data = self._data ** a
            
        return self.__class__(data=data)

    def __neg__(self):
        data = -self._data
        return self.__class__(data=data)

    def __pos__(self):
        data = self._data
        return self.__class__(data=data)

    def abs(self):
        return _VariableND_Base(self._data.abs())
    
    
    def reduce_min(self):
        return self._data.reduce_min()
    
    def reduce_minabs(self):
        return self._data.reduce_minabs()
    
    def reduce_max(self):
        return self._data.reduce_max()
    
    def reduce_maxabs(self):
        return self._data.reduce_maxabs()
    
    
    def set(
        self,
        a
    ):
        if isinstance(a, _VariableND_Base):
            assert self._data.shape == a._data.shape, "Shape mismatch"
            self._data = a._data
        
        elif isinstance(a, np.ndarray):
            assert self.shape == a.shape, "Shape mismatch: "+str(self.shape)+" != "+str(a.shape)
            self._data = libpdefd_array.array(a)
            
        elif isinstance(a, libpdefd_array._array_base):
            assert self.shape == a.shape, "Shape mismatch: "+str(self.shape)+" != "+str(a.shape)
            self._data = libpdefd_array.array(a)
            
        else:
            raise Exception("Type '"+str(type(a))+"' not supported")

        assert isinstance(self._data, libpdefd_array._array_base)

    
    def set_all(
        self,
        value
    ):
        self._data.set_all(value)


    def to_numpy_array(self):
        """
        Return numpy array
        """
        assert isinstance(self._data, libpdefd_array._array_base)
        return self._data.to_numpy_array()

    def __str__(self):
        retstr = ""
        retstr += "variable: "+str(self._name)
        retstr += ", shape: "+str(self.shape)
        assert self.shape == self._data.shape
        return retstr



def VariableND(
        data_or_grid = None,
        name = None,
    ):
        
    retval = _VariableND_Base()
    
    retval._name = name
    
    if isinstance(data_or_grid, GridInfoND) or isinstance(data_or_grid, GridInfo1D):
        retval._data = libpdefd_array.array_zeros(data_or_grid.shape)

    elif isinstance(data_or_grid, libpdefd_array._array_base):
        retval._data = libpdefd_array.array(data)
        
    else:
        raise Exception("Type '"+str(type(data_or_grid))+" not supported")

    assert isinstance(retval._data, libpdefd_array._array_base)
    
    retval.shape = retval._data.shape
    
    return retval

"""

class VariableND(_VariableND_Base):
    def __init__(self, *args, **kwargs):
        _VariableND_Base.__init__(self)
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        grid,
        name = None,
        data = None
    ):
        assert isinstance(grid, GridInfoND), "Only GridInfoND supported for grid parameter"
        
        self._name = name        
        self.grid = grid
        self._data = data
        self.shape = self.grid.shape
        
        if self._data is None:
            self._data = libpdefd_array.array_zeros(self.shape)
            #self._data = np.zeros(self.shape)
    
        assert isinstance(self._data, libpdefd_array._array_base)
    
            
    def __str__(self):
        retstr = ""
        retstr += "variable: "+str(self._name)
        retstr += ", shape: "+str(self.shape)
        return retstr
"""


def VariableNDSet_Empty_Like(variable_set_like: list):
    return VariableNDSet([None for _ in range(len(variable_set_like.variable_list))])



class VariableNDSet:
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    
    def setup(
        self,
        variable_list: list,
    ):
        if not isinstance(variable_list, list):
            if not isinstance(variable_list, Variable1D) and not isinstance(variable_list, VariableND):
                raise Exception("Only Variables are supported")
            
            variable_list = [variable_list]

        self.variable_list = variable_list
    
    
    def copy(self):
        v = VariableNDSet([i.copy() for i in self.variable_list])
        return v
    
    
    def __getitem__(self, key):
        
        if isinstance(key, str):
            for i in range(len(self.variable_list)):
                if self.variable_list[i]._name == key:
                    return self.variable_list[i]
                
            raise Exception("Field '"+str(key)+"' not found in set of variables")
        
        return self.variable_list[key]
    
    
    def __setitem__(self, key, data):
        
        if isinstance(key, str):
            for i in range(len(self.variable_list)):
                if self.variable_list[i]._name == key:
                    self.variable_list[i] = data
                    return
            
            raise Exception("Field '"+str(key)+"' not found in set of variables")
        
        self.variable_list[key] = data
    
    
    def set(
        self,
        variable_list_or_class
    ):
        if isinstance(variable_list_or_class, list):
            self.variable_list[:] = variable_list_or_class[:]
        
        elif isinstance(variable_list_or_class, VariableNDSet):
            self.variable_list[:] = variable_list_or_class.variable_list[:]
                
        else:
            raise Exception("TODO")
    
    
    def __add__(self, a):
        return VariableNDSet([self.variable_list[i] + a.variable_list[i] for i in range(len(a.variable_list))])
    
    def __sub__(self, a):
        return VariableNDSet([self.variable_list[i] - a.variable_list[i] for i in range(len(a.variable_list))])
    
    def __mul__(self, a):
        if isinstance(a, _VariableND_Base):
            return VariableNDSet([self.variable_list[i] * a.variable_list[i] for i in range(len(self.variable_list))])
 
        else:
            return VariableNDSet([self.variable_list[i] * a for i in range(len(self.variable_list))])
    
    
    def __rmul__(self, a):
        if isinstance(a, Variable1D):
            return VariableNDSet([a.variable_list[i] * self.variable_list[i] for i in range(len(self.variable_list))])
 
        else:
            return VariableNDSet([a * self.variable_list[i] for i in range(len(self.variable_list))])


    def __div__(self, a):
        return VariableNDSet([self.variable_list[i] / a.variable_list[i] for i in range(len(a.variable_list))])

    def __iadd__(self, a):
        for i in range(len(self.variable_list)):
            self.variable_list[i] += a.variable_list[i]
        return self

    def __isub__(self, a):
        for i in range(len(self.variable_list)):
            self.variable_list[i] -= a.variable_list[i]
        return self

    def __imul__(self, a):
        for i in range(len(self.variable_list)):
            self.variable_list[i] *= a.variable_list[i]
        return self

    def __idiv__(self, a):
        for i in range(len(self.variable_list)):
            self.variable_list[i] /= a.variable_list[i]
        return self

    def __neg__(self):
        return VariableNDSet([-v for v in self.variable_list])

    def __pos__(self):
        return VariableNDSet([v for v in self.variable_list])

    def __str__(self):
        retstr = "VariableNDSet:\n"
        for i in range(len(self.variable_list)):
            retstr += " + "+str(self.variable_list[i]._name)+": shape="+str(self.variable_list[i]._data.shape)+"\n"
        
        return retstr

