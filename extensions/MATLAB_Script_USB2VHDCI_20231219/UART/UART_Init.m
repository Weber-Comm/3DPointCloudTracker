clc;
delete(instrfindall);
%COM_Port_Name='com4';
if (isempty(instrfind({'Port', 'Type'}, {upper(COM_Port_Name), 'serial'})))
    COM=serial(COM_Port_Name);
end
set(COM,'BaudRate',115200,'DataBits',8,'StopBits',1,'Parity','none','FlowControl','none');
set(COM,'BytesAvailableFcnMode', 'byte');
COM.InputBufferSize = 128;
COM.OutputBufferSize = 128;
COM.BytesAvailableFcnCount = 8;
%COM.BytesAvailableFcn=@UART_CallBack;
%configureCallback(COM,"byte",1,@UART_CallBack);

COM.Timeout= 1;
%COM.Terminator = 'LF';%'CR'
UART.Head=[81 170 90];%[0x51,0xAA,0x5A]
if(strcmp(COM.Status,'closed'))
    fopen(COM);
end

% COM
%
