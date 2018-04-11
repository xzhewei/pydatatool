import os

def mkdir_if_missing(path):
    """
    If path not exist, then mkdir.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False

def save_pkl(data,filename):
    """
    Save a data to a .pkl file
    """
    import cPickle
    with open(filename,'w') as f:
        cPickle.dump(data,f)

def load_pkl(filename):
    """
    Load a data from a .pkl file.
    """
    import cPickle
    with open(filename,'r') as f:
        data = cPickle.load(f)
    return data

def load_json(filename):
    import json
    A = json.load(open(filename,'r'))
    return A

def save_json(data,filename):
    import json
    path, _ = os.path.split(filename)
    mkdir_if_missing(path)
    json_file = open(filename,'w')
    json.dump(data, json_file)
    json_file.close()
