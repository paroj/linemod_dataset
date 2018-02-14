"""
utilites to read the linemod dataset to numpy types

@author Pavel Rojtberg <https://github.com/paroj>
"""

import numpy as np

def ply_vtx(path):
    """
    read all vertices from a ply file
    """
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()

    N = int(f.readline().split()[-1])

    while f.readline().strip() != "end_header":
        continue

    pts = []

    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))

    return np.array(pts)

def linemod_pose(path, i):
    """
    read a 3x3 rotation and 3x1 translation.
    
    can be done with np.loadtxt, but this is way faster
    """
    R = open("{}/data/rot{}.rot".format(path, i))
    R.readline()
    R = np.float32(R.read().split()).reshape((3, 3))

    t = open("{}/data/tra{}.tra".format(path, i))
    t.readline()
    t = np.float32(t.read().split())
    
    return R, t

def linemod_dpt(path):
    """read a depth image"""
    dpt = open(path, "rb")
    rows = np.frombuffer(dpt.read(4), dtype=np.int32)[0]
    cols = np.frombuffer(dpt.read(4), dtype=np.int32)[0]
    
    return np.fromfile(dpt, dtype=np.uint16).reshape((rows, cols))