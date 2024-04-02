clear all; %#ok<CLALL>
% Create a TCP/IP server that listens on port 12345. Replace '12345' with the port number you are using.
server = tcpserver("::", 12345);

% Display a message to indicate the server is ready and listening for a connection.
disp('Server is ready and waiting for a connection...');

% Loop indefinitely to accept and process connections
while true
    % Wait for a connection
    while server.Connected == 0
        pause(0.1); % Pause to reduce CPU usage
    end
    disp('Client connected.');

    % Process incoming data from the connected client
    while server.Connected > 0
        if server.NumBytesAvailable > 0
            dataReceived = read(server, server.NumBytesAvailable, 'string');
            % Read the data sent by the client
            % dataReceived = readline(server); % Reads a line of text
            
            % Check if the received data is not empty
            if dataReceived ~= ""
                % Decode the JSON formatted string to MATLAB data
                dataDecoded = jsondecode(dataReceived);
                
                % Display or process the decoded data
                % disp('Received data:');
                % disp(dataDecoded);
                disp([char(datetime), ' Received: ', mat2str(dataDecoded, 4)]);
                
                if size(dataDecoded,1) > 1
                    warning("This frame is considered invalid because multiple objects detected, ")
                end
                % Here, decodedData is a MATLAB array that you can
                % further process depending on your application needs.
                % For example, plotting bounding boxes on an image.
            end
        else
            pause(0.01); % Pause to reduce CPU usage
        end
    end
    
    disp('Client disconnected.');
end
 

% Clean up by deleting the server object when done.
% This step might not be reached if the loop is designed to run indefinitely.
% Consider adding a condition to break the loop and reach this code.
delete(server);
disp('Server closed.');
