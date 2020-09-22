#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:49:27 2019

@author: rain
"""

import numpy as np

DataDir="/media/rain/Win10_F/KITTI_odometry/Dataset/sequences/00/"
iFrame=0

PC=np.fromfile(str(DataDir+"velodyne/"+str(iFrame).zfill(6)+".bin"), dtype=np.float32, count=-1).reshape([-1,4])



fo = open(str(DataDir+str(iFrame).zfill(6)+".txt"), "w")
for iRow in range(PC.shape[0]):
    fo.write(str(str(PC[iRow,0])+" "+str(PC[iRow,1])+" "+str(PC[iRow,2])+"\n") )
    
fo.close()



