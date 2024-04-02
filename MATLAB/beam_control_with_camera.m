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

% global tx ty tz; % Measure the relative position of the antennas and  RGB camera
% tx = 0;
% ty = 0;
% tz = 0;

global t_theta t_phi;
t_theta = 0; % Input in deg
t_phi = 0; % vertical angle is not necessary 

global latestBoundingBox; %#ok<*GVMIS>
latestBoundingBox = 0;

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
    global latestBoundingBox;
    global hasReceivedData;
    global is_newest;
    persistent server;

    if isempty(server)
        server = tcpserver("::", 12345);
    end

    if server.NumBytesAvailable > 0
        dataReceived = read(server, server.NumBytesAvailable, 'string');
        if dataReceived ~= ""
            % Decode the JSON formatted string to MATLAB data
            dataDecoded = jsondecode(dataReceived);
            
            % Display or process the decoded data
            % disp('Received data:');
            % disp(dataDecoded);
            % disp([char(datetime), ' Received: ', mat2str(dataDecoded, 4)]);
            disp([datestr(datetime('now')) ', received bounding box: ' mat2str(dataDecoded, 4)]);
            
            if size(dataDecoded,1) == 1
                
                latestBoundingBox = dataDecoded; %calculation
           
                hasReceivedData = true;
                is_newest = true;
                
            elseif size(dataDecoded,1) == 0
                warning("This frame is considered invalid due to no detected object.")
            else 
                warning("This frame is considered invalid due to multiple detected objects.")
            end
        end
            
            
    end
end

function callBeamSet(~, ~)
    global latestBoundingBox;
    global hasReceivedData;
    global is_newest;
    persistent isRunning;
    
    if isempty(isRunning)
        isRunning = false;
    end

    if hasReceivedData && ~isRunning && is_newest
        isRunning = true; %#ok<*NASGU>
        try
            beamSet(latestBoundingBox);
        catch e
            disp(['Error calling beamSet: ', e.message]);
        end
        is_newest = false;
        isRunning = false;
    end
end

function beamSet(bbox)
    global t_theta t_phi;
    global B;
    global PSA;
    global COM; %#ok<NUSED>

    % Calculate relative position

    % pass
    
    % Bounding box to angle
    HFOV = 90;
    VFOV = 67.5;
    theta = calculateAngleFromBoundingBox(bbox(1),bbox(3),HFOV);
    phi = calculateAngleFromBoundingBox(bbox(1),bbox(3),VFOV);
    
    % Modify the sign according to whether the coordinate system is a left-handed or right-handed system.
    theta = theta + t_theta; %!!!!!!!!!!!!
    phi = phi + t_phi;
    
    % Find the beam index H

    theta = mod(theta + 180, 360) - 180;
    theta = - theta;
    [~, index] = min(abs(B(:,2) - theta)); % Nearest theta
    beam_index = B(index, 1);

    %
    
    disp([datestr(datetime('now')) ', theta: ' num2str(theta) ', phi: ' num2str(phi)]);
    if abs(theta)<60
        disp([datestr(datetime('now')) ', Beam setting successful with beam index ' num2str(beam_index)]);
    else
        disp([datestr(datetime('now')) ', angle exceeds the boundary with beam index: ' num2str(beam_index)]);
    end
    
    tic;
    PSA.OpMode=13;
    PSA.BeamH=beam_index;
    PSA.BeamV=0;
    PSA.TXON=1;
    PSA.RXON=0;
    SetmmPSA_TR64A;

    

end




