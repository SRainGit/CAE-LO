clc;
clear;
close all;

Pts = zeros(201,3);
Pts(:,1) = -100:1:100;

Pts_ = CorrectPts(Pts, 22);

figure, hold on;
plot3(Pts(:,1),Pts(:,2),Pts(:,3),'r.');
plot3(Pts_(:,1),Pts_(:,2),Pts_(:,3),'b.');
axis equal
rotate3d on