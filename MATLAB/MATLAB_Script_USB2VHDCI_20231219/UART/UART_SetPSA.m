% tic;
Generate_CtrlFrame;
CtrlPacket;
UART.CheckSum=rem(sum([UART.Head CtrlPacket]),256);
UART_Frame=[UART.Head CtrlPacket UART.CheckSum];
fwrite(COM,UART_Frame,'uint8');
UART_Back=(fread(COM,8,'uint8')).';

PSA.Temp_code=UART_Back(6)*256+UART_Back(7);
PSA.Temp_deg=(PSA.Temp_code/1024*3.3*1000-776)/2.86;
% disp(sprintf('Temperature is %.2f °„C\n', PSA.Temp_deg));
% toc;