clear;

% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% Please Mannually Run delete(timerfind) after User Terminate
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

global latestXYZ;
latestXYZ = [0, 0, 0];

global hasReceivedData;
hasReceivedData = false;

global is_newest;
is_newest = false;

try
    % Receiver Loop
    t1 = timer('ExecutionMode', 'fixedRate', 'Period', 0.005, 'TimerFcn', @receiveData);
    start(t1);
    disp([datestr(datetime('now')) ', start 1'])

    % Call beamSet Loop
    t2 = timer('ExecutionMode', 'fixedRate', 'Period', 0.005, 'TimerFcn', @callBeamSet);
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
            disp([datestr(datetime('now')) ', received data X: ' num2str(x) ', Y: ' num2str(y) ', Z: ' num2str(z)]);
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
        isRunning = true;
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
    disp([datestr(datetime('now')) ', beamSet started with X: ' num2str(x) ', Y: ' num2str(y) ', Z: ' num2str(z)]);
    
    % Simulate beamSet operation by pausing for 0.5 seconds
    pause(0.5);

    disp([datestr(datetime('now')) ', beamSet end']);
end
