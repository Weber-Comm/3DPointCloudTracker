% General Control Frame Generation
% By ZK 2021.02.26


% CtrlElement = [23,25,0,1,0,1,1,1,183] 
% CtrlElementLength = [4,6,1,1,1,1,1,1,6] 


function CtrlBody = GenerateFrame(CtrlElement,CtrlElementLength)


if (size(CtrlElementLength)~=size(CtrlElement))
    disp('Element length is not the same!')
    CtrlBody = 0;
    return    
end
    

sz=size(CtrlElementLength);
CtrlStr=[];
for i=1:sz(1,2)

    if (CtrlElement(i)<0) 
       CtrlElement(i)= 0;
    elseif CtrlElement(i) > (2^CtrlElementLength(i)-1)
       CtrlElement(i)=2^CtrlElementLength(i)-1;
    end
   

     myElement=CtrlElement(i);
     
     tempbinstr=dec2bin(myElement,1); 
     
  
     while length(tempbinstr) < CtrlElementLength(i)
          tempbinstr = strcat('0',tempbinstr);
     end
     
     CtrlStr = strcat(CtrlStr,tempbinstr);
      
end



if (length(CtrlStr)~=sum(CtrlElementLength))

    disp('ctrl length is error!')
    CtrlBody = 0;
    return 
end


if mod(sum(CtrlElementLength),8)>0
    j=8-mod(sum(CtrlElementLength),8);
    while j>0 
    CtrlStr= strcat('0',CtrlStr);
    j=j-1;
    end
end


CtrlBody=[];
for j=1:length(CtrlStr)/8
    tmpstrstart=(j-1)*8+1;
    tmpstrstop=tmpstrstart+7;
    tmpstr=CtrlStr(tmpstrstart:tmpstrstop);
    CtrlBody(j)=bin2dec(tmpstr);
end

end

