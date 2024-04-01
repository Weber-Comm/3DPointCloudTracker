%Applied-Wave Inc
%detect connected device
addpath('.\UART');
addpath('.\General');

%%
COM_Port_Name='com3';
UART_Init;

%%

% Update Beam EXT-TR
PSA.OpMode=14;
PSA.BeamH=12;
PSA.BeamV=10;

SetmmPSA_TR64A;

%%
%Update Beam INT_TR
PSA.OpMode=13;
PSA.BeamH=0;
PSA.BeamV=0;
PSA.TXON=1;
PSA.RXON=0;
SetmmPSA_TR64A;

%%
%ANT BLOCK SELECT MODE
PSA.OpMode=12;
PSA.ANTSEL = [1 1 1 1,1 1 1 1,1 1 1 1,1 1 1 1];
PSA.FBS_ID =7;
SetmmPSA_TR64A;





