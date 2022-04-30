#-----------------------------------------------------------------------------------------------------
# ftypes.py
#-----------------------------------------------------------------------------------------------------


class TEXT:
    def save(path, data):
        with open(path, 'wt') as f:
            f.write(data)
        return path
    def load(path):
        with open(path, 'rt') as f:
            data = f.read()
        return data


class JSON:
    import json
    def save(path, data):
        """ saves a dict to disk in json format """
        with open(path, 'w') as f:
            f.write(JSON.json.dumps(data, sort_keys=False, indent=4))
        return path
    def load(path):
        """ returns a dict from a json file """
        data = None
        with open(path, 'r') as f:
            data = JSON.json.loads(f.read())
        return data


class NP:
    from numpy import load
    from numpy import save


class IM:
    from matplotlib.pyplot import imread as load
    from matplotlib.pyplot import imsave as save


class FIG:
    save = lambda path, data: data.savefig(path)
    load = lambda path: path


class ITOR:
    from known.ds.struct import iTOR
    save = iTOR.save
    load = iTOR.load


