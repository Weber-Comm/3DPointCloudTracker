clear all;
delete(timerfind);
% Client Side Script to receive t_theta from the server

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

% Continuously receive data
while true
    if client.NumBytesAvailable > 0
        data = read(client, client.NumBytesAvailable, 'string');
        disp(['Received t_theta: ' data]);
    end
    pause(0.1); % Pause to reduce CPU usage, adjust as needed
end

