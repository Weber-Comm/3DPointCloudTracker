% 读取CSV文件
data = readmatrix('beam_index.csv');

% 输入角度值theta
theta = input('请输入角度值theta: ');

% 标准化输入角度到[-180, 180]区间
theta = mod(theta + 180, 360) - 180;

% 计算与输入角度最接近的角度值，并找到对应的波束序号
[~, index] = min(abs(data(:,2) - theta));

% 输出最接近的波束序号
beamIndex = data(index, 1);
fprintf('最接近的波束序号是: %d\n', beamIndex);