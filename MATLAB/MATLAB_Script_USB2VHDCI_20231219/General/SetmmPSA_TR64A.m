%tic
CtrlHead=[81 170 90]; %0x51 0xAA 0x5A

if ( PSA.OpMode==14)
    CtrlData=[0 PSA.BeamH PSA.BeamV 0 PSA.OpMode];
    CtrlLength=[12 6 6 4 4]; 
end

if ( PSA.OpMode==13)
    CtrlData=[0 PSA.BeamH PSA.BeamV 0 PSA.TXON PSA.RXON PSA.OpMode];
    CtrlLength=[12 6 6 2 1 1 4]; 
end


if ( PSA.OpMode==12)
    CtrlData=[0 PSA.ANTSEL PSA.FBS_ID  PSA.OpMode];
    CtrlLength=[8 ones(1,16) 4 4]; 
end

if ( PSA.OpMode==5)
    CtrlData=[PSA.CycleTimeSet_ms PSA.CodeBookUsed PSA.CycleToGo 0 PSA.OpMode];
    CtrlLength=[8 8 8 4 4]; 
end

if ( PSA.OpMode==6)
    CtrlData=[PSA.CodeAddr PSA.BeamH PSA.BeamV 0 PSA.OpMode];
    CtrlLength=[8 8 8 4 4]; 
end


if ( PSA.OpMode==7 || PSA.OpMode==8)
    CtrlData=[hex2dec('FE') hex2dec('FD') hex2dec('FC') 0 PSA.OpMode];
    CtrlLength=[8 8 8 4 4]; 
end

CtrlBody = GenerateFrame(CtrlData,CtrlLength);
a=sum([CtrlHead CtrlBody]);
CheckSUM=rem(sum([CtrlHead CtrlBody]),256);
UART_Send=[CtrlHead CtrlBody CheckSUM];


dec2hex(UART_Send);
                
% tic               
fwrite(COM,UART_Send,'uint8');
% toc
UART_Back=(fread(COM,8,'uint8')).'
% toc

PSA.Temp_Code=UART_Back(6)*256+UART_Back(7);
PSA.Temp_deg=(PSA.Temp_Code/1024*3.3*1000-776)/2.86;
% sprintf('Temperature is %.2f°„C',PSA.Temp_deg)


    