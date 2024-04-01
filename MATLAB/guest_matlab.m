clear all;

% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% Please Mannually Run delete(timerfind) after User Terminate
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

addpath('.\MATLAB_Script_USB2VHDCI_20231219\General');
addpath('.\MATLAB_Script_USB2VHDCI_20231219\UART');

%% 

global COM
global PSA

COM_Port_Name='com3';
UART_Init;

% ANT BLOCK SELECT MODE
PSA.OpMode=12;
PSA.ANTSEL = [1 1 1 1,1 1 1 1,1 1 1 1,1 1 1 1];
PSA.FBS_ID =7;
SetmmPSA_TR64A;

%%

global B;
B = readmatrix('beam_index.csv');

global tx ty tz; % Measure the relative position of the antennas and LiDAR
tx = 0;
ty = 0;
tz = 0;

global t_theta t_phi;
t_theta = -60; % Input in deg
t_phi = 0; % vertical angle is not necessary 

global latestXYZ; %#ok<*GVMIS>
latestXYZ = [0, 0, 0];

global hasReceivedData;
hasReceivedData = false;

global is_newest;
is_newest = false;

try
    % Receiver Loop
    t1 = timer('ExecutionMode', 'fixedRate', 'Period', 0.01, 'TimerFcn', @receiveData);
    start(t1);
    disp([datestr(datetime('now')) ', start 1']) %#ok<*DATST>

    % Call beamSet Loop
    t2 = timer('ExecutionMode', 'fixedRate', 'Period', 0.01, 'TimerFcn', @callBeamSet);
    start(t2);
    disp([datestr(datetime('now')) ', start 2'])

catch e
    disp(['!!!ERROR!!!: ', e.message]);
   
    delete(timerfind)

end

function receiveData(~, ~)
    global latestXYZ;
    global hasReceivedData;
    global is_newest;
    persistent server;

    if isempty(server)
        server = tcpserver("::", 54322);
    end

    if server.NumBytesAvailable > 0
        data = read(server, server.NumBytesAvailable, 'string');
        coords = strsplit(data, ',');
        if length(coords) == 3
            x = str2double(coords{1});
            y = str2double(coords{2});
            z = str2double(coords{3});
            latestXYZ = [x, y, z];
           
            hasReceivedData = true;
            is_newest = true;

            disp([datestr(datetime('now')) ', received position X: ' num2str(x) ', Y: ' num2str(y) ', Z: ' num2str(z)]);
            
        end
    end
end

function callBeamSet(~, ~)
    global latestXYZ;
    global hasReceivedData;
    global is_newest;
    persistent isRunning;
    
    if isempty(isRunning)
        isRunning = false;
    end

    if hasReceivedData && ~isRunning && is_newest
        isRunning = true; %#ok<*NASGU>
        try
            beamSet(latestXYZ(1), latestXYZ(2), latestXYZ(3));
        catch e
            disp(['Error calling beamSet: ', e.message]);
        end
        is_newest = false;
        isRunning = false;
    end
end

function beamSet(x, y, z)
    global tx ty tz;
    global t_theta t_phi;
    global B;
    global PSA;
    global COM; %#ok<NUSED>

    % rigid transformation

    [xp, yp, zp] = rigidTransform(x, y, z, tx, ty, tz, 0, 0, 0);

    % Cartesian to spherical

    [~, theta, phi] = cartesianToSpherical(xp, yp, zp); % Deg
    % Modify the sign according to whether the coordinate system is a left-handed or right-handed system.
    theta = theta + t_theta; %!!!!!!!!!!!!
    phi = phi + t_phi;
    
    % Find the beam index H

    theta = mod(theta + 180, 360) - 180;
    theta = - theta;
    [~, index] = min(abs(B(:,2) - theta)); % Nearest theta
    beam_index = B(index, 1);

    %
    
    disp([datestr(datetime('now')) ', transformed position X: ' num2str(xp) ', Y: ' num2str(yp) ', Z: ' num2str(zp)]);
    disp([datestr(datetime('now')) ', theta: ' num2str(theta) ', phi: ' num2str(phi)]);
    if abs(theta)<60
        warning([datestr(datetime('now')) ', beamSet started with beam index: ' num2str(beam_index)]);
    else
        warning([datestr(datetime('now')) ', angle exceeds the boundary with beam index: ' num2str(beam_index)]);
    end
    
    tic;
    PSA.OpMode=13;
    PSA.BeamH=beam_index;
    PSA.BeamV=0;
    PSA.TXON=1;
    PSA.RXON=0;
    SetmmPSA_TR64A;

    disp([datestr(datetime('now')) ', Beam setting successful with beam index ' num2str(beam_index) ', time taken' num2str(toc) 'seconds.']);

end




