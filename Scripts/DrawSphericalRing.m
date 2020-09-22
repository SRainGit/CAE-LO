
function DrawSphericalRing(Radius, LatitudeStart, LatitudeStep, LatitudeEnd, LongitudeStart, LongitudeStep, LongitudeEnd)

LatitudeRange = LatitudeStart:LatitudeStep:LatitudeEnd;
LongitudeRange = LongitudeStart:LongitudeStep:LongitudeEnd;

LatitudeRange = LatitudeRange*pi/180;
LongitudeRange = LongitudeRange*pi/180;


[theta,phi] = meshgrid(LongitudeRange,LatitudeRange);
[x,y,z] = sph2cart(theta, phi, Radius);
surf(x,y,z)


end