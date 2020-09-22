#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:27:44 2019

@author: rain
"""

import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import dot as dot

from Transformations import *


def GetErrorEulers(relRs0, relRs1):
    assert relRs0.shape[0] == relRs1.shape[0]
    nDatas = relRs0.shape[0]    
    ErrorEulers = np.zeros((nDatas,3), dtype=np.float32)
    for i in range(nDatas):
        errorR = dot(np.linalg.inv(relRs0[i,:,:]), relRs1[i,:,:])
        eulers = RotateMat2EulerAngle_XYZ(errorR)
        ErrorEulers[i,:] = eulers        
    return ErrorEulers
    
    

def GetErrorRTs(poses, poses_, Tr, isPlot):        
    # ----- then compare with the ground truth
    relRs, relTs, relEulers, diffNormRelEulers, diffNormRelTs = GetLidarDiffRels(poses, Tr)
    relRs_, relTs_, relEulers_, diffNormRelEulers_, diffNormRelTs_ = GetLidarDiffRels(poses_, Tr)
    
    errorRelEulers = GetErrorEulers(relRs, relRs_)
    errorRelTs = relTs_ - relTs
    errorRelEulersNorm = LA.norm(errorRelEulers, axis=1)
    errorRelTsNorm = LA.norm(errorRelTs, axis=1)
    
    if isPlot == True:
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.title('relEulers')
        plt.plot(relEulers,'.')
        plt.subplot(2, 4, 2)
        plt.title('relTs')
        plt.plot(relTs,'.')
        plt.subplot(2, 4, 3)
        plt.title('diffNormRelEulers')
        plt.plot(diffNormRelEulers,'.')
        plt.subplot(2, 4, 4)
        plt.title('diffNormRelTs')
        plt.plot(diffNormRelTs,'.')
        plt.subplot(2, 4, 5)
        plt.title('relEulers_')
        plt.plot(relEulers_,'.')
        plt.subplot(2, 4, 6)
        plt.title('relTs_')
        plt.plot(relTs_,'.')
        plt.subplot(2, 4, 7)
        plt.title('diffNormRelEulers_')
        plt.plot(diffNormRelEulers_,'.')
        plt.subplot(2, 4, 8)
        plt.title('diffNormRelTs_')
        plt.plot(diffNormRelTs_,'.')
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('errorRelEulersNorm')
        plt.plot(errorRelEulersNorm,'.')
        plt.subplot(1, 2, 2)
        plt.title('errorRelTsNorm')
        plt.plot(errorRelTsNorm,'.')
        
#        plt.figure()
#        plt.subplot(1, 3, 1)
#        plt.title('errorRelEulers-X')
#        plt.plot(errorRelEulers[:,0],'.')
#        plt.subplot(1, 3, 2)
#        plt.title('errorRelEulers-Y')
#        plt.plot(errorRelEulers[:,1],'.')
#        plt.subplot(1, 3, 3)
#        plt.title('errorRelEulers-Z')
#        plt.plot(errorRelEulers[:,2],'.')
#        
#        plt.figure()
#        plt.subplot(1, 3, 1)
#        plt.title('errorRelTs-X')
#        plt.plot(errorRelTs[:,0],'.')
#        plt.subplot(1, 3, 2)
#        plt.title('errorRelTs-Y')
#        plt.plot(errorRelTs[:,1],'.')
#        plt.subplot(1, 3, 3)
#        plt.title('errorRelTs-Z')
#        plt.plot(errorRelTs[:,2],'.')
        
        plt.show
        
    
    GroundTruthRels = []
    GroundTruthRels.append(relRs)
    GroundTruthRels.append(relTs)
    GroundTruthRels.append(relEulers)
    GroundTruthRels.append(diffNormRelEulers)
    GroundTruthRels.append(diffNormRelTs)
    
    EstimatedRels = []
    EstimatedRels.append(relRs_)
    EstimatedRels.append(relTs_)
    EstimatedRels.append(relEulers_)
    EstimatedRels.append(diffNormRelEulers_)
    EstimatedRels.append(diffNormRelTs_)
    
    return GroundTruthRels, EstimatedRels, errorRelEulers, errorRelTs
    

    
    
    
    
    
    
    

