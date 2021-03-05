for i=1:length(Tinput)
 Toutput(i).out=(sin(Tinput(i).Xrv1))+(Tinput(i).parameterA)*(sin(Tinput(i).Xrv2).^2)+(Tinput(i).parameterB)*((Tinput(i).Xrv3)^4)*(sin(Tinput(i).Xrv1));
    
    
    
end