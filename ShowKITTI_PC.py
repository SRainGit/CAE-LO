#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:37:34 2019

@author: rain
"""

import numpy as np
import mayavi.mlab
 
 
#pointcloud = np.fromfile(str("/media/rain/Win10_F/KITTI_object_3D/object/training/velodyne/000265.bin"), dtype=np.float32, count=-1).reshape([-1,4])
#pointcloud = np.fromfile(str("/media/rain/Win10_F/KITTI_odometry/data_odometry_velodyne/dataset/sequences/01/velodyne/000498.bin"), dtype=np.float32, count=-1).reshape([-1,4])
pointcloud = np.fromfile(str("/media/rain/Win10_F/KITTI_odometry/Dataset/sequences/00/velodyne/000001.bin"), dtype=np.float32, count=-1).reshape([-1,4])


print(pointcloud.shape)
x = pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point
r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
 
 
vals='height'
if vals == "height":
    col = z
else:
    col = d
 
fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
mayavi.mlab.points3d(x, y, z,
                     col,          # Values used for Color
                     mode="point",
                     colormap='spectral', # 'bone', 'copper', 'gnuplot'
                     #color=r,
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )
 
x=np.linspace(5,5,50)
y=np.linspace(0,0,50)
z=np.linspace(0,5,50)
mayavi.mlab.plot3d(x,y,z)
mayavi.mlab.show()
