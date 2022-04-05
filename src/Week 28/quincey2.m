clear all;
close all;

n=17;

a0=0.42;
b=0;

c=0;    
for l=1:100000;
    X=zeros(1,n);
    
    X(1)=randn(1)*sqrt(1/(1-a0^2));
    for k=2:n
        X(k)=b+a0*(X(k-1)-b)+randn(1);
    end;
    
    %LOOCV parameter estimates
    
    %%%approx b
    
    hat_b=(sum(X)-X)/n;
    hat_a=zeros(size(hat_b));
    CV_0=zeros(size(hat_b));
    CV_1=zeros(size(hat_b));
    for k=2:(n-1)
        hat_a(k)=((X(2:n)-hat_b(k))*(X(1:(n-1))-hat_b(k))'-(X(k)-hat_b(k))*(X(k-1)+X(k+1)-2*hat_b(k)))/(sum((X(2:n)-hat_b(k)).^2)-(X(k)-hat_b(k))^2-(X(k+1)-hat_b(k))^2);
        CV0(k)=(X(k)-hat_b(k)-a0*(X(k-1)-hat_b(k)))^2;
        CV1(k)=(X(k)-hat_b(k)-hat_a(k)*(X(k-1)-hat_b(k)))^2;
    end;
    
    c=c+(sum(CV0(2:(n-1)))<sum(CV1(2:(n-1))));
end;
%probability of doing it right...
c/100000
    
