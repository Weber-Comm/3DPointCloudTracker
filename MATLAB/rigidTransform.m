function [xp, yp, zp] = rigidTransform(x, y, z, tx, ty, tz, theta_x, theta_y, theta_z)

    Rx = [1, 0, 0, 0;
          0, cos(theta_x), -sin(theta_x), 0;
          0, sin(theta_x), cos(theta_x), 0;
          0, 0, 0, 1];

    Ry = [cos(theta_y), 0, sin(theta_y), 0;
          0, 1, 0, 0;
          -sin(theta_y), 0, cos(theta_y), 0;
          0, 0, 0, 1];

    Rz = [cos(theta_z), -sin(theta_z), 0, 0;
          sin(theta_z), cos(theta_z), 0, 0;
          0, 0, 1, 0;
          0, 0, 0, 1];

    T = [1, 0, 0, tx;
         0, 1, 0, ty;
         0, 0, 1, tz;
         0, 0, 0, 1];

    Transform = T * Rx * Ry * Rz;

    P = [x; y; z; 1];
    P_prime = Transform * P;

    xp = P_prime(1);
    yp = P_prime(2);
    zp = P_prime(3);
end
