
function R=EulerAngle2RotateMat(angX,angY,angZ,RotateSequnce)

R=eye(3,3);
R_X=[1 0 0; 0 cos(angX) -sin(angX); 0 sin(angX) cos(angX)];
R_Y=[cos(angY) 0 sin(angY); 0 1 0; -sin(angY) 0 cos(angY)];
R_Z=[cos(angZ) -sin(angZ) 0; sin(angZ) cos(angZ) 0; 0 0 1];

for i=1:3
    if RotateSequnce(i)=='x' || RotateSequnce(i)=='X'
        R=R_X*R;
    elseif RotateSequnce(i)=='y' || RotateSequnce(i)=='Y'
        R=R_Y*R;
    elseif RotateSequnce(i)=='z' || RotateSequnce(i)=='Z'
        R=R_Z*R;
    else
        errordlg('Error!');
    end
end

end