clear all;
delete(timerfind);
% 初始化全局变量
global theta_r;
theta_r = 0; 


t1 = timer('ExecutionMode', 'fixedRate', 'Period', 0.5, 'TimerFcn', @sendData);
start(t1);


% 数据发送函数
function sendData(~, ~)
    global theta_r;
    persistent server;

    if isempty(server)
        server = tcpserver("127.0.0.1", 33333);
        disp(['Server initialized on port 33333. Waiting for client connections...']);
    end

    if server.Connected
        try
            write(server, num2str(theta_r), "string");
            disp([datestr(datetime('now')), ', Sent theta_r: ', num2str(theta_r)]);
        catch e
            disp(['Error sending data: ', e.message]);
        end
    end
    % 更新 t_theta 的值
    theta_r = theta_r + 0.1;
end
