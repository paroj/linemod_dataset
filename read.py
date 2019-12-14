"""
utilites to read the linemod dataset to numpy types

@author Pavel Rojtberg <https://github.com/paroj>
"""

import numpy as np

def _parse_ply_header(f):
    assert f.readline().strip() == "ply"
    vtx_count = 0
    idx_count = 0

    for l in f:
        if l.strip() == "end_header":
            break
        elif l.startswith("element vertex"):
            vtx_count = int(l.split()[-1])
        elif l.startswith("element face"):
            idx_count = int(l.split()[-1])
    
    return vtx_count, idx_count

def ply_vtx(path):
    """
    read all vertices from a ply file
    """
    f = open(path)
    vtx_count = _parse_ply_header(f)[0]

    pts = []

    for _ in range(vtx_count):
        pts.append(f.readline().split()[:3])

    return np.array(pts, dtype=np.float32)

def ply_idx(path):
    """
    read all indices from a ply file
    """
    f = open(path)
    vtx_count, idx_count = _parse_ply_header(f)
    
    for _ in range(vtx_count):
        f.readline()
    
    idx = []
    for _ in range(idx_count):
        idx.append(f.readline().split()[1:4])

    return np.array(idx, dtype=np.int32)

def transform(path):
    """
    read the to-origin transform.dat
    
    @return R, t in [cm]
    """
    f = open(path)
    f.readline() # 12
    
    T = []
    for l in f:
        T.append(l.split()[1])
    
    T = np.float32(T).reshape((3, 4))
    return T[:3, :3], T[:, 3] * 100 # [m] > [cm]

def linemod_pose(path, i):
    """
    read a 3x3 rotation and 3x1 translation.
    
    can be done with np.loadtxt, but this is way faster
    @return R, t in [cm]
    """
    R = open("{}/data/rot{}.rot".format(path, i))
    R.readline()
    R = np.float32(R.read().split()).reshape((3, 3))

    t = open("{}/data/tra{}.tra".format(path, i))
    t.readline()
    t = np.float32(t.read().split())
    
    return R, t

def linemod_dpt(path):
    """
    read a depth image
    
    @return uint16 image of distance in [mm]"""
    dpt = open(path, "rb")
    rows = np.frombuffer(dpt.read(4), dtype=np.int32)[0]
    cols = np.frombuffer(dpt.read(4), dtype=np.int32)[0]
    
    return np.fromfile(dpt, dtype=np.uint16).reshape((rows, cols))