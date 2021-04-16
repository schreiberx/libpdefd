from libpdefd.core.gridinfo import *


class MeshND:
    """
    Create mesh from grid
    """
    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        grid_info_nd: GridInfoND,
        name = None
    ):
        self.grid_info_nd = grid_info_nd
        self.name = name
        
        self.data = grid_info_nd.get_mesh()
        
        if name is None:
            self.name = grid_info_nd.name



class MeshNDSet:
    name_counter = 0

    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
    
    def setup(
        self,
        mesh_or_grid_nd_list : list,
        name = None
    ):
        self.name = name
        
        if self.name is None:
            self.name = "mesh_nd_set_"+str(MeshNDSet.name_counter)
            MeshNDSet.name_counter += 1
        

        if isinstance(mesh_or_grid_nd_list[0], GridInfoND):
            """
            Greate MeshNDSet based on GridInfoND
            """
            _grid_info_nd_list = [MeshND(i) for i in mesh_or_grid_nd_list]
            self.mesh_nd_list = MeshNDSet(_grid_info_nd_list)
        
        else:
            """
            MeshNDSet already exists
            """
            self.mesh_nd_list = mesh_or_grid_nd_list
            for i in range(len(self.mesh_nd_list)):
                assert isinstance(self.mesh_nd_list[i], MeshND)
    
    
    def __getitem__(self, key):
        
        if isinstance(key, str):
            for i in range(len(self.mesh_nd_list)):
                if self.mesh_nd_list[i].name == key:
                    return self.mesh_nd_list[i]
                
            raise Exception("Field '"+str(key)+"' not found in set of mesh sets")
        
        return self.mesh_nd_list[key]
    
    
    def __setitem__(self, key, data):
        
        if isinstance(key, str):
            for i in range(len(self.mesh_nd_list)):
                if self.mesh_nd_list[i].name == key:
                    self.mesh_nd_list[i] = data
                    return
            
            raise Exception("Field '"+str(key)+"' not found in set of grid infos")
        
        self.mesh_nd_list[key] = data


    def __len__(self):
        return len(self.grid_info_nd_list)

    def __str__(self):
        retstr = "MeshNDSet: "+self.name+"\n"
        for i in range(len(self.mesh_nd_list)):
            retstr += " + "+str(self.mesh_nd_list[i].name)+": shape="+str(self.mesh_nd_list[i].data.shape)+"\n"
        
        return retstr
