# CAE-LO
[CAE-LO: LiDAR Odometry Leveraging Fully Unsupervised Convolutional Auto-Encoder for Interest Point Detection and Feature Description](https://arxiv.org/abs/2001.01354)
```
@article{yin2020caelo,
    title={CAE-LO: LiDAR Odometry Leveraging Fully Unsupervised Convolutional Auto-Encoder for Interest Point Detection and Feature Description},
    author={Deyu Yin and Qian Zhang and Jingbin Liu and Xinlian Liang and Yunsheng Wang and Jyri Maanpää and Hao Ma and Juha Hyyppä and Ruizhi Chen},
    journal={arXiv preprint arXiv:2001.01354},
    year={2020}
}
@article{
    title={Interest Point Detection from Multi-Beam LiDAR Point Cloud Using Unsupervised CNN},
    author={Deyu Yin, Qian Zhang, Jingbin Liu, Xinlian Liang, Yunsheng Wang, Shoubin Chen, Jyri Maanpää, Juha Hyyppä, Ruizhi Chen},
    journal={IET Image Processing},
    year={2020}
}
```
![image](https://github.com/SRainGit/CAE-LO/blob/master/Docs/CAE-LO%20method%20overview.png)

See the rankings in [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), our method's name is "CAE-LO".


# Usage
1. Basic enviornments for python3 and Keras. Simple networks. No worries. Package requirements can be found in `requirements.txt`.
2. `Dirs.py` to modify dictionaries.
3. `BatchProcess.py` projects PCs to spherical rings with multi-thread processing.
4. `BatchVoxelization.py` project PCs into multi-solution voxel model with multi-thread processing.
5. `SphericalRing.py` to do basic function about spherical ring model, especially the function of getting keypts.
6. You can try `Match.py` to see some demos using trained models.
7. `PoseEstimation.py` to generate initial odometry.
8. `RefinePoses.py` to generate refined odometry based on extended interest points & ground normals, and also to show generated trajectories. (The code for generating ground normals is currently commented. Uncomment it if you want to use.)


# Notes
1. Generated interest points and features for sequence 00 and 01 can be found in [GoogleDrive](https://drive.google.com/open?id=1MATZrnTgBXeKmaIyC-x5dRHrZ6hX9Hl0).
2. The final refined trajectories of sequence 00-10 can be found in [GoogleDrive](https://drive.google.com/file/d/1bayF_HDgZHo4gpu0vAIHNrjQhN1JHog-/view?usp=sharing).
3. The data arragement format is simple. Just folders like "KeyPts", "Features", "InliersIdx", "SphericalRing", etc.
4. Package PCLKeypoint in `PclKeyPts.py` can be installed from: https://github.com/lijx10/PCLKeypoints.
5. If you have any problems or confunsions, please post them in ISSUES or contact me by email.
