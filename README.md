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


# Pre-release
Thank you all for the interest. Pre-release-1.

For the newers, I suggest you to wait for my formal release.


# Usage
1. Basic enviornments for python3 and Keras. Simple networks. No worries. Package requirements can be found in `requirements.txt`.
2. `Dirs.py` to modify dictionaries.
3. `BatchProcess.py` to do batch processings on projecting PC to spherical rings and getting keypts by using multi-thread.
4. `BatchVoxelization.py` to project PC into multi-solution voxel model and basic functions about multi-resolution model.
5. `SphericalRing.py` to do basic function about spherical ring model, importantly for getting keypts by using trained model and cupy.
6. You can try `Match.py` to see some demos using trained models
7. `PoseEstimation.py` to generate initial odometry.
8. `RefinePoses.py` to generate refined odometry based on extended interest points and ground normals. (The code for generating ground normals is currently commented. Uncomment it if you want to use.)
9. There are some other test codes I tried to find other solutions. You may get some new ideas I hope. I will remove them in the formal release.


# Notes
1. Generated interest points and features for sequence 00 and 01 can be found in [GoogleDrive](https://drive.google.com/open?id=1MATZrnTgBXeKmaIyC-x5dRHrZ6hX9Hl0).
2. The extracting voxel patches part is slow, I will try to do some optimization.
3. The data arragement format is simple. Just serveral folders of "KeyPts", "Features", "InliersIdx", "SphericalRing", etc.
4. If you have any problems or confunsions, please post them in ISSUES or contact me by email.
