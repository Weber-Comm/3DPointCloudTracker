function [r, theta, phi] = cartesianToSpherical(x, y, z)

    r = sqrt(x^2 + y^2 + z^2);
    
    % 计算方位角theta，范围是[-pi, pi]
    theta = atan2(y, x); % 使用atan2以处理x=0的情况
    
    % 计算极角phi，范围是[0, pi]
    phi = atan2(sqrt(x^2 + y^2), z); % 或使用acos(z / r)

    theta = rad2deg(theta);
    phi = rad2deg(phi);
end