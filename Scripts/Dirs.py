#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:29:20 2019

@author: rain
"""

#strBaseDir = '/media/rain/Win10_F/KITTI_odometry/'
#str3DFeatNetDir = '/media/rain/Win10_F/KITTI_odometry/output_3DFeatNet/'
#strUsipBaseDir = '/media/rain/Win10_F/USIP/'

strBaseDir = 'F:\\KITTI_odometry\\'
str3DFeatNetDir = 'F:\\KITTI_odometry\\output_3DFeatNet\\'
strUsipBaseDir = 'F:\\USIP\\'


strGroundTruthPosesDir = strBaseDir + 'poses/'
strEstimatedPosesDir = strBaseDir + 'poses_/'
strDejumpyedPosesDir = strBaseDir + 'poses__/'
strRefinedPosesDir = strBaseDir + 'poses___/'
#strRefinedPosesDir = strBaseDir + 'poses___WithoutBackwardUpdate/'
strClosedPosesDir = strBaseDir + 'poses____/'

strDataBaseDir = strBaseDir + 'velodyne/sequences/'
strCalibDataDir = strBaseDir + 'calib/'




strUsipPcDir = strUsipBaseDir + '/kitti/data_odometry_velodyne/numpy/'
#strUsipKeyPtsDir = strUsipBaseDir + 'KeyPts/kitti/tsf/'
strUsipKeyPtsDir = strUsipBaseDir + 'KeyPts/kitti/tsf_1024/'
strUsipDescDir = strUsipBaseDir + 'Descriptors/'