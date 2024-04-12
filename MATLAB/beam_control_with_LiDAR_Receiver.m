clear all;
delete(timerfind);

% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% Please Mannually Run delete(timerfind) after User Terminate
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

addpath('.\MATLAB_Script_USB2VHDCI_20231219\General');
addpath('.\MATLAB_Script_USB2VHDCI_20231219\UART');

global COM %#ok<*GVMIS>
global PSA

% COM_Port_Name='com3';
% UART_Init;

% ANT BLOCK SELECT MODE
% PSA.OpMode=12;
% PSA.ANTSEL = [1 1 1 1,1 1 1 1,1 1 1 1,1 1 1 1];
% PSA.FBS_ID =7;
% SetmmPSA_TR64A;
global is_theta_newest;
is_theta_newest = false;

global theta_r;
theta_r = NaN; 

global t_theta;
t_theta = 0; % Input in deg
global t_phi;
t_phi = 0; % Vertical angle is not necessary

global B;
B = readmatrix('beam_index.csv');

% Setup the connection
host = "127.0.0.1"; 
port = 33333;

try
    client = tcpclient(host, port);
    disp('Connected to the server.');
catch e
    disp(['Error connecting to server: ', e.message]);
    return;
end

t_recv = timer('ExecutionMode', 'fixedRate', 'Period', 0.001, 'TimerFcn', {@receiveData, client});
start(t_recv);

t_proc = timer('ExecutionMode', 'fixedRate', 'Period', 0.1, 'TimerFcn', @processData);
start(t_proc);

function receiveData(~, ~, client)
    global theta_r;
    global is_theta_newest;

    if client.NumBytesAvailable > 0
        data = read(client, client.NumBytesAvailable, 'string');
        disp(['Received theta_r: ' char(data)]);
        theta_r = str2double(data);
        is_theta_newest = true;
    end
end

function processData(~, ~)
    global theta_r;
    global t_theta;
    global t_phi;
    global is_theta_newest;
    global B;

    if is_theta_newest == true && ~isnan(theta_r)
        
        is_theta_newest = false;

        phi = 0;

        theta = theta_r + t_theta;
        phi = phi + t_phi;

        theta = mod(theta + 180, 360) - 180;
        theta = -theta;
        [~, index] = min(abs(B(:,2) - theta));
        beam_index = B(index, 1);

        disp([datestr(datetime('now')) ', theta: ' num2str(theta) ', phi: ' num2str(phi)]);
        if abs(theta) < 60
            disp([datestr(datetime('now')) ', Beam setting successful with beam index ' num2str(beam_index)]);
        else
            disp([datestr(datetime('now')) ', angle exceeds the boundary with beam index: ' num2str(beam_index)]);
        end
        
        % tic;
        % PSA.OpMode=13;
        % PSA.BeamH=beam_index;
        % PSA.BeamV=0;
        % PSA.TXON=1;
        % PSA.RXON=0;
        % SetmmPSA_TR64A;
        pause(0.2);
        % 在使用波束控制代码的时候，请把占位用的pause(0.2);注释掉！！！！！！！！
        
        

    end
end


