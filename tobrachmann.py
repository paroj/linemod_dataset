"""
convert the data for
"Uncertainty-Driven 6D Pose Estimation of Objects and Scenes from a Single RGB Image"

@author: Pavel Rojtberg <https://github.com/paroj>
"""

import os.path
import glob
import numpy as np
import cv2

import read

# convert from the lineMOD coordinate system to the OpenCV CS
t_LM2cv    = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# convert from OpenCV coordinate sytem to the CS used by brachmann
t_cv2brach = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

# info.txt template: all units in [m]
info = """image size
640 480
{}
rotation:
{}
center:
{}
extent:
{}
"""

def write_info(path, obj_name, R, t, extent, ctr):
    """
    write brachmann info file, given the linemod data
    @param t: distance to center in [m]
    @param extent: extent in [mm]
    @param ctr: center of bounding box in [mm] 
    """
    # bbox center from view
    ctr = R.dot(ctr) / 10  # [mm] > [cm]
    R, t = lm2brach(R, t + ctr) # brachmann stores distance to center
    
    # write info file
    t /= 100 # [cm] > [m]
    
    extent = t_cv2brach.dot(extent)/1000 # [mm] > [m]
    
    f = open(path, "w")
    
    R_str = "\n".join([" ".join(r) for r in R.astype(str)])
    t_str = " ".join(t.astype(str))
    ext_str = " ".join(extent.astype(str))
    
    txt = info.format(obj_name, R_str, t_str, ext_str)

    f.write(txt)

def lm2brach(R, t):
    """
    convert linemod to brachmann pose format
    """
    R = t_LM2cv.dot(R)
    t = t_LM2cv.dot(t)
    
    # result may be reconstructed behind the camera
    if cv2.determinant(R) < 0:
        R *= -1
        t *= -1
    
    return t_cv2brach.dot(R.T).T, t # TODO why dont we transfrom t as well?

def main():
    # all non hidden directories
    objs = [d for d in sorted(os.listdir(".")) if os.path.isdir(d) and d[0] not in (".", "_")]
    
    for o in objs:
        pts = read.ply_vtx("{}/mesh.ply".format(o)) # [mm]
        omin = np.min(pts, axis=0)
        omax = np.max(pts, axis=0)
        
        octr = (omin + omax) * 0.5
        extent = omax - omin
    
        N = len(glob.glob("{}/data/color*.jpg".format(o)))
    
        infop = "{}/info".format(o)
        depthp = "{}/depth_noseg".format(o)
        rgbp = "{}/rgb_noseg".format(o)
        segp = "{}/seg".format(o)
        objp = "{}/obj".format(o)
        
        for p in [infop, depthp, rgbp, segp, objp]:
            if not os.path.exists(p):
                os.mkdir(p)
    
        for i in range(N):
            dpt = read.linemod_dpt("{}/data/depth{}.dpt".format(o, i))
            cv2.imwrite("{}/depth_{:05d}.png".format(depthp, i), dpt)
            
            # just jpg > png, doh!
            color = cv2.imread("{}/data/color{}.jpg".format(o, i))        
            cv2.imwrite("{}/color_{:05d}.png".format(rgbp, i), color)
            
            R, t = read.linemod_pose(o, i)
            write_info("{}/info_{:05d}.txt".format(infop, i), o, R, t, extent, octr)

if __name__ == "__main__":
    main()
