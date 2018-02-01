# Converter Utilities for the ACCV LINEMOD Dataset

The LINEMOD dataset is widely used for various 6D pose estimation and camera localization algorithms.

Unfortunately each author chose to convert the original data into an own file format and only support loading from that data.

This repository therefore provides a set of python functions to read and convert the original LINEMOD data to the respective formats.

This way one can compare the algorithms using the exact same input data.

## ACCV LINEMOD Dataset

The extensive ACCV database is public (over 18000 real images with 15 different objects and ground truth pose)! 

Each dataset contains the 3D model saved as a point cloud object.xyz
format: 
```
#_of_voxels size_of_voxel_in_cm 
x1_in_cm y1_in_cm z1_in_cm normal_x1 normal_y1 normal_z1 color_x1_normalized_to_1 color_y1_normalized_to_1 color_z1_normalized_to_1
...
```
and a file called distance.txt with the maximum diameter of the object (in cm). 

For some datasets we also provide a nice mesh model in the ply format (in mm - with better normals). The original mesh is contained in OLDmesh.ply. For most datasets we registered this OLDmesh.ply to the point cloud with the transformation stored in transform.dat (first number is not important, then each first number of a line is obsolete - for the rest: the transformation matrix [R|T] is stored rowwise (in m)). 

The registered mesh is stored in mesh.ply. In the folder data you can find the color images, the aligned depth images and the ground truth rotation and translation (in cm).

In order to read the depth images you can use this function. 

The internal camera matrix parameters for the kinect are:
```
fx=572.41140, px=325.26110, fy=573.57043, py=242.04899
```

Color image and depth image are already aligned by the internal alignement procedure of Kinect.
