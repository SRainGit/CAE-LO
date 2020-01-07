
EvaluationResults.mat is a 9x7 matrix. Each row shows one frame to frame matching result and each column repreents on perfermence item. The reseaon of why it has 9 rows please see our paper in the experiment part. But the orders of the rows in mat file and in the paper are different.


Set the methods are:
0, ours; 1, 3DFeatNet; 2, USIP


Each method has two parts: interetest point detection and descriptor


from row 1 to row 9:
0+0;
0+1;
0+2;
1+0;
1+1;
...
2+1;
2+2;


from column 1 t0 column 7:
RRE, stdRRE, RTE, stdRTE, success rate, inlier ratio, average iterations
