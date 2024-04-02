function angle = calculateAngleFromBoundingBox(bbox1, bbox2, FOV)


bbox_center = (bbox1 + bbox2) / 2;

offset_from_center_normalized = bbox_center - 0.5;

angle = rad2deg(atan(2*offset_from_center_normalized*tan(deg2rad(FOV)/2)));



end