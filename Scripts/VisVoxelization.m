

function [OriPt, idxPatchPosition, idxVoxels, VoxelFaces, idxPatchVoxels, PatchVoxelFaces] = ...
    VisVoxelization(OriPC, VoxelSize, PatchPosition, PatchSize)
HalfPatchSize = PatchSize/2;

%% Voxelization
% shift to non-negative zone
mins = min(OriPC,[],1);
PC = OriPC - mins;

% get indexes of voxels
idxVoxels = floor(PC/VoxelSize); 

idxes = idxVoxels(:,1)*10000*10000 + idxVoxels(:,2)*10000 + idxVoxels(:,3);
sortedData = [idxVoxels, idxes, PC];
sortedData = sortrows(sortedData,4);

diff = sortedData(2:size(idxes,1),4)-sortedData(1:size(idxes,1)-1,4);
diff = [1;diff];
idxVoxels = sortedData(diff>0,1:3);
sortedPC = sortedData(diff>0,5:7);

nVoxels = size(idxVoxels,1)


%% Build Faces
% build the meta-face at first
MetaFaces = zeros(6,4,3);  % 6 faces; each face has 4 points
for iAxis = 1:3
    iAxis1 = mod((iAxis+1)-1,3) + 1;
    iAxis2 = mod((iAxis+2)-1,3) + 1;
    for iDirection = 0:1
        iFace = (iAxis-1)*2 + iDirection + 1;
        % reset iVertice, because it's a new face
        iVertice = 0;
        for iSide1 = 0:1
            for iSide2 = 0:1
                iVertice = iVertice + 1;
                MetaFaces(iFace, iVertice, iAxis) = iDirection;
                MetaFaces(iFace, iVertice, iAxis1) = iSide1;
                MetaFaces(iFace, iVertice, iAxis2) = iSide2;
            end
        end
    end
end
% adjust to the formal order of dimensions
MetaFaces(:, [3, 4], :) = MetaFaces(:, [4, 3], :);
MetaFaces = permute(MetaFaces,[2,1,3]);

% generate all the faces based on the voxel indexes
AllFaces = repmat(MetaFaces,1,1,1,nVoxels);
AllFaces = permute(AllFaces,[1,2,4,3]);

% generate all the idxVoxels to be added to the meta faces
idxVoxels_Faces = repmat(idxVoxels,1,1,4,6);
idxVoxels_Faces = permute(idxVoxels_Faces,[3,4,1,2]);

% shift the AllFaces by the idxes of voxels
AllFaces = AllFaces + idxVoxels_Faces;
VoxelFaces = reshape(AllFaces, size(AllFaces,1), size(AllFaces,2)*size(AllFaces,3), size(AllFaces,4));



%% Get the voxel patch
PatchPosition = PatchPosition - mins;
idxPatch = floor(PatchPosition/VoxelSize); 
idxPatchPosition = idxPatch;

distVs = idxVoxels - idxPatch;
dists = vecnorm(distVs, 2, 2);
[minVal, minIdx] = min(dists);
OriPt = sortedPC(minIdx,:) + mins;

distX = idxVoxels(:,1) - idxPatch(:,1);
distY = idxVoxels(:,2) - idxPatch(:,2);
distZ = idxVoxels(:,3) - idxPatch(:,3);

% idxPatchVoxels = find(distX>-HalfPatchSize & distX<=HalfPatchSize);
idxPatchVoxels = find(distX>-HalfPatchSize & distX<=HalfPatchSize & ...
    distY>-HalfPatchSize & distY<=HalfPatchSize & ...
    distZ>-HalfPatchSize & distZ<=HalfPatchSize);
idxPatchVoxels = idxVoxels(idxPatchVoxels,:);
size(idxPatchVoxels,1)

% generate all the faces based on the voxel indexes
AllFaces = repmat(MetaFaces,1,1,1,size(idxPatchVoxels,1));
AllFaces = permute(AllFaces,[1,2,4,3]);

% generate all the idxVoxels to be added to the meta faces
PatchVoxelFaces = repmat(idxPatchVoxels,1,1,4,6);
PatchVoxelFaces = permute(PatchVoxelFaces,[3,4,1,2]);

% shift the AllFaces by the idxes of voxels
AllFaces = AllFaces + PatchVoxelFaces;
PatchVoxelFaces = reshape(AllFaces, size(AllFaces,1), size(AllFaces,2)*size(AllFaces,3), size(AllFaces,4));


end














