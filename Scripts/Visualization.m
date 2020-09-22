clc; clear; close all;

fip = fopen('F:\KITTI_odometry\velodyne\sequences\00\velodyne\000000.bin','rb'); 
% fip = fopen('/media/rain/Win10_F/KITTI_odometry/velodyne/sequences/00/velodyne/000000.bin','rb'); 
[data,num] = fread(fip,'float32');

% PC = reshape(data, [size(data,1)/4,4]);
PC = reshape(data, [4,size(data,1)/4])';

figure(1), hold on
h = [-1100 700 1000 900];
set(gcf,'Position',h)
grid off, axis off
axis equal

% title('PC')
pcshow(PC(:, 1:3), [0, 0, 1], 'MarkerSize', 5)

Radius = 80;
LatitudeStart = -24.8;
LatitudeStep = 1;
LatitudeEnd = 4;
LongitudeStart = 0;
LongitudeStep = 2;
LongitudeEnd = 360;
% DrawSphericalRing(Radius, LatitudeStart, LatitudeStep, LatitudeEnd, LongitudeStart, LongitudeStep, LongitudeEnd)


xRange = [-70, 70];
yRange = [-70, 70];
zRange = [-15, 15];
step = 10;
xStep = step;
yStep = step;
zStep = step;

% DrawGrid(xRange,yRange,zRange,xStep,yStep,zStep)


%#set(gca, 'Box','on', 'LineWidth',2, 'XTick',x, 'YTick',y, 'ZTick',z, ...
%#  'XLim',[x(1) x(end)], 'YLim',[y(1) y(end)], 'ZLim',[z(1) z(end)])
%#xlabel x, ylabel y, zlabel z

view(3), axis vis3d, rotate3d on
% camproj perspective, 






PatchPosition = [3.5, 8.0, -1.65];
PatchSize = 16;
HalfPatchSize = PatchSize/2;

VoxelSizes = [0.02, 0.16, 0.64];
ViewRanges = [2, 4, 12];
for iSize = 3:3
    VoxelSize = VoxelSizes(iSize);
    [OriPt, idxPatch, idxVoxels, VoxelFaces, idxPatchVoxels, PatchVoxelFaces] = ...
        VisVoxelization(PC(:,1:3), VoxelSize, PatchPosition, PatchSize);
    OriPt

    figure(), hold on
    patch(VoxelFaces(:,:,1),VoxelFaces(:,:,2),VoxelFaces(:,:,3), 'b')
    patch(PatchVoxelFaces(:,:,1),PatchVoxelFaces(:,:,2),PatchVoxelFaces(:,:,3), 'r')

    VoxelRange = [idxPatch-HalfPatchSize+1; idxPatch+HalfPatchSize+1];
    DrawGrid(VoxelRange(:,1),VoxelRange(:,2),VoxelRange(:,3),1,1,1)
    
    axis off
    view(3), axis vis3d, rotate3d on, axis equal
    
%     ViewRange = ViewRanges(iSize);
%     xlim([idxPatch(1) - ViewRange/VoxelSize, idxPatch(1) + ViewRange/VoxelSize])
%     ylim([idxPatch(2) - ViewRange/VoxelSize, idxPatch(2) + ViewRange/VoxelSize])
%     zlim([idxPatch(3) - ViewRange/VoxelSize, idxPatch(3) + ViewRange/VoxelSize])
end






