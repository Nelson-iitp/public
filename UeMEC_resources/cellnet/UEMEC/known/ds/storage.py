#-----------------------------------------------------------------------------------------------------
# storage.py
#-----------------------------------------------------------------------------------------------------
import os, json


#------------------------------------
# Storage 
#------------------------------------
class STORAGE:
    DEFAULT_ROOT_STORE = '.' #<--- this store refers to the root dir as a store, the relative path is blank
    """
        Represents a collection of stores

        [Members]
        self.cwd        abs path of current working directory (of python)
        self.storage    abs path of storage config (json) file
        self.root       abs path of root dir which is the dir containing storage file
                        Note: 'root_relative_paths' are relative to this root
        self.stores     a dictionary of 'store_alias' v/s 'root_relative_paths' <-- contents of storage file

    """
    def create(root, storage, auto_load=True, auto_save=True):
        """
        creates a new STORAGE with root dir at 'root' where 'storage' is the name of storage configuration file
        * prefered method of creating new storage from scratch
        """
        return STORAGE( os.path.join(root, storage), auto_create=True, auto_load=auto_load, auto_save=auto_save )

    def __init__(self, storage, auto_create=True, auto_load=True, auto_save=True):
        """
        Create a storage configuration using a json file 'storage'
        storage can be relative or absolute, only absolute paths are stored
        storage ia the abs path for storage file
        use STORAGE.create(root, storage) if 'storage' is just the name of storage file located inside root
        """

        self.cwd = os.getcwd() #<-- always returns a abs path
        self.storage = os.path.abspath(storage) # stores abs paths in instances
        self.root = os.path.dirname(self.storage) # abs dir name <-- root
        self.autosave = auto_save

        self.create_config() if auto_create else None #<-- create if not existing, it has no effect if storage exists
        self.load_config() if auto_load else None # without calling load_config(), self.stores will not be created
        
    def load_config(self):
        if os.path.exists(self.storage):
            with open(self.storage, 'r') as f:
                self.stores = json.loads(f.read())
            return True
        else:
            return False

    def save_config(self):
        if os.path.exists(self.storage):
            with open(self.storage, 'w') as f:
                f.write(json.dumps(self.stores, sort_keys=False, indent=4))
            return True
        else:
            return False

    def create_config(self):
        os.makedirs(self.root, exist_ok=True) 
        if not os.path.exists(self.storage):
            with open(self.storage, 'w') as f:
                f.write(json.dumps({STORAGE.DEFAULT_ROOT_STORE:''}, sort_keys=False, indent=4))

    def register(self, **stores):
        """ register multiple stores using a dict of "alias":"root_relative_path """
        for alias, root_rel_path in stores.items():
            self.stores[alias] = root_rel_path
            full_path =  os.path.join(self.root, root_rel_path)
            os.makedirs( full_path, exist_ok=True )
        return self.save_config() if self.autosave else None #<=== save json after adding


    #------------------------------------
    # path query 
    #------------------------------------
    def path_rel(self, alias, file):
        return os.path.join(self.stores[alias], file) 

    def path_abs(self, alias, file):
        return os.path.join(os.path.join(self.root, self.stores[alias]), file)

    def path_py(self, alias, file):
        return os.path.relpath(os.path.join(os.path.join(self.root, self.stores[alias]), file), self.cwd)

    def paths(self, alias, file): # path_ds replaced by call
        """ returns 3-tuple (r,a,p) paths where
            r = root realtive path
            a = absolute path
            p = python relative path
        """
        root_rel_path = self.stores[alias] #<--- ds path relative to root
        file_root_rel_path = os.path.join(root_rel_path, file) #<--- file path relative to root

        root_abs_path = os.path.join(self.root, root_rel_path) #<--- ds path abs
        file_root_abs_path = os.path.join(root_abs_path, file) #<--- abs file path

        py_rel_path = os.path.relpath(file_root_abs_path, self.cwd)

        return file_root_rel_path, file_root_abs_path, py_rel_path # relp, absp, pyp

    def __call__(self, alias, file):  #<--- returns python relative path
        return self.path_py(alias, file)     


    #------------------------------------
    # file IO 
    #------------------------------------
    def save(self, alias, file, data, saveF):
        """ save a file in this storage using saveF function like (lambda path, data: None) """
        r,a,p = self.paths(alias, file)
        saveF(p, data)
        return r, a, p

    def save_py(self, alias, file, data, saveF):
        """ save a file in this storage using saveF function like (lambda path, data: None) """
        p = self.path_py(alias, file)
        saveF(p, data)
        return p

    def save_abs(self, alias, file, data, saveF):
        """ save a file in this storage using saveF function like (lambda path, data: None) """
        a = self.path_abs(alias, file)
        saveF(a, data)
        return a

    def load(self, alias, file, loadF):
        """ load a file from this storage using loadF function like (lambda path: data) """
        r,a,p = self.paths(alias, file)
        return loadF(p), r, a, p

    def load_py(self, alias, file, loadF):
        """ load a file from this storage using loadF function like (lambda path: data) """
        p = self.path_py(alias, file)
        return loadF(p), p

    def load_abs(self, alias, file, loadF):
        """ load a file from this storage using loadF function like (lambda path: data) """
        a = self.path_abs(alias, file)
        return loadF(a), a



#------------------------------------
# Forage
#------------------------------------
class FORAGE(STORAGE):
    """ uses ftype to save data, ftype is something that implements load and save methods 
        save = lambda path, data : None
        load = lambda path       : data
    """

    def create(root, storage, auto_load=True, auto_save=True):
        """
        creates a new STORAGE with root dir at 'root' where 'storage' is the name of storage configuration file
        * prefered method of creating new storage from scratch
        """
        return FORAGE( os.path.join(root, storage), auto_create=True, auto_load=auto_load, auto_save=auto_save )
        
    def save(self, alias, file, data, ftype):
        r,a,p = self.paths(alias, file)
        ftype.save(p, data)
        return r, a, p

    def save_py(self, alias, file, data, ftype):
        p = self.path_py(alias, file)
        ftype.save(p, data)
        return p

    def save_abs(self, alias, file, data, ftype):
        a = self.path_abs(alias, file)
        ftype.save(a, data)
        return a

    def load(self, alias, file, ftype):
        r,a,p = self.paths(alias, file)
        return ftype.load(p), r, a, p

    def load_py(self, alias, file, ftype):
        p = self.path_py(alias, file)
        return ftype.load(p), p

    def load_abs(self, alias, file, ftype):
        a = self.path_abs(alias, file)
        return ftype.load(a), a


#------------------------------------
# CTFORAGE
#------------------------------------
class CFORAGE(STORAGE):
    def __init__(self, storage, ftype, auto_create=True, auto_load=True, auto_save=True):
        super().__init__(storage, auto_create, auto_load, auto_save)
        self.ftype=ftype

    def create(root, storage, ftype, auto_load=True, auto_save=True):
        """
        creates a new STORAGE with root dir at 'root' where 'storage' is the name of storage configuration file
        * prefered method of creating new storage from scratch
        """
        return CFORAGE( os.path.join(root, storage), ftype=ftype, auto_create=True, auto_load=auto_load, auto_save=auto_save )

    def save(self, alias, file, data):
        r,a,p = self.paths(alias, file)
        self.ftype.save(p, data)
        return r, a, p

    def save_py(self, alias, file, data):
        p = self.path_py(alias, file)
        self.ftype.save(p, data)
        return p

    def save_abs(self, alias, file, data):
        a = self.path_abs(alias, file)
        self.ftype.save(a, data)
        return a

    def load(self, alias, file):
        r,a,p = self.paths(alias, file)
        return self.ftype.load(p), r, a, p

    def load_py(self, alias, file):
        p = self.path_py(alias, file)
        return self.ftype.load(p), p

    def load_abs(self, alias, file):
        a = self.path_abs(alias, file)
        return self.ftype.load(a), a
#-----------------------------------------------------------------------------------------------------
# Foot-Note:
""" NOTE:
    * Author:           Nelson.S
"""
#-----------------------------------------------------------------------------------------------------
