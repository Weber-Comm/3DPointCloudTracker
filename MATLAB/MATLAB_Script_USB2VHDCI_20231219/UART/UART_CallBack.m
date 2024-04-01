function UART_CallBack(obj,event)
[UART_Rx,a]=fread(obj,1,'uint8');
end