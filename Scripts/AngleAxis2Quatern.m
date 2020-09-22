function q = AngleAxis2Quatern(angle, axisVector)
    q = zeros(4,1);
    halfAngle = angle/2;
    sinHalfAngle = sin(halfAngle);
    q(1) = cos(halfAngle);
    q(2) = axisVector(1)*sinHalfAngle;
    q(3) = axisVector(2)*sinHalfAngle;
    q(4) = axisVector(3)*sinHalfAngle   ; 
end