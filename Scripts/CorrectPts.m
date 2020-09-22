
function Pts_ = CorrectPts(Pts, Angle)
    CalibAngle = Angle * pi / 180;
    Pts_ = Pts;
    
    zAxis = zeros(size(Pts,1),3);
    zAxis(:,3) = zAxis(:,3) + 1;
    
    rotVs = cross(Pts, zAxis);
    for iPt = 1:size(Pts,1)
        pt = Pts_(iPt,:);
        v = rotVs(iPt,:)/norm(rotVs(iPt,:));
        q = AngleAxis2Quatern(CalibAngle, v);
        R = Quatern2RotMat(q);
        Pts_(iPt,:) = (R*pt')';
    end
    Pts_(isnan(Pts_)) = 0;
end