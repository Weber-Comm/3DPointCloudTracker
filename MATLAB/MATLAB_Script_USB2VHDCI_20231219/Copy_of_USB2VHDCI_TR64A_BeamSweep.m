%Applied-Wave Inc
%detect connected device
addpath('.\UART');
addpath('.\General');

%%
COM_Port_Name='com3';
UART_Init;

%% 
% Set CycleToGo , CodeBookUsed, and CycleTimeSet_ms
PSA.OpMode=5;
PSA.CycleToGo=0; %0~255,0表示一直循环，其他为循环次数
PSA.CodeBookUsed=127; %0~127,表示循环使用的码本最大下标；
PSA.CycleTimeSet_ms=50 ; % ms
SetmmPSA_TR64A;

%% 
% Set Codebook Addr=0~127
BeamH=[];BeamV=[];

for i=0:1:63
    index=i+1;
    BeamH(index)=i;
    BeamV(index)=0;
end


for i=64:1:127
    index=i+1;
    BeamH(index)=i-64;
    BeamV(index)=1;
end

for i=0:1:127
    PSA.OpMode=6;
    PSA.CodeAddr=i;
    index=i+1;
    PSA.BeamH=BeamH(index);
    PSA.BeamV=BeamV(index);
    fprintf("=======CodeAddr=%d===========",i)
    SetmmPSA_TR64A;
    pause(0.1)
end


%% 
%Start Beam Sweep
PSA.OpMode=7;
SetmmPSA_TR64A;


%% 
%Stop Beam Sweep
%any UART Command will terminate the Beam Sweep
PSA.OpMode=8;
SetmmPSA_TR64A;


%%
%test1
PSA.OpMode=6;
PSA.BeamH=0;
PSA.BeamV=1;
PSA.CodeAddr=0;
SetmmPSA_TR64A;

pause(0.1)

PSA.OpMode=5;
PSA.CycleToGo=1; %0~255,0表示一直循环，其他为循环次数
PSA.CodeBookUsed=1; %0~127,表示循环使用的码本最大下标；
PSA.CycleTimeSet_ms =100 ; % ms
SetmmPSA_TR64A;

pause(0.1)

PSA.OpMode=7;
SetmmPSA_TR64A;


%%
%test2
BeamH=[];BeamV=[];

for i=0:1:63
    index=i+1;
    BeamH(index)=i;
    BeamV(index)=0;
end


for i=64:1:127
    index=i+1;
    BeamH(index)=i-64;
    BeamV(index)=1;
end

for i=0:1:127
    PSA.OpMode=6;
    PSA.CodeAddr=i;
    index=i+1;
    PSA.BeamH=BeamH(index);
    PSA.BeamV=BeamV(index);
    fprintf("=======CodeAddr=%d===========",i)
    SetmmPSA_TR64A;
    pause(0.1)
end

pause(0.1)

PSA.OpMode=5;
PSA.CycleToGo=0; %0~255,0表示一直循环，其他为循环次数
PSA.CodeBookUsed=127; %0~127,表示循环使用的码本最大下标；
PSA.CycleTimeSet_ms=25 ; % ms
SetmmPSA_TR64A;

pause(0.1)

PSA.OpMode=7;
SetmmPSA_TR64A;


