


function PlotCube(Position, MetaFaces)

Faces(:,:,1) = MetaFaces(:,:,1) + Position(1);
Faces(:,:,2) = MetaFaces(:,:,2) + Position(2);
Faces(:,:,3) = MetaFaces(:,:,3) + Position(3);

fill3(Faces(:,:,1), Faces(:,:,2), Faces(:,:,3), 'blue')

end