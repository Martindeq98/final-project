function [B,P]=sudoku(A,P);

if nargin==1;
    P=zeros(81,9);
    a=A(:);
    for k=1:81;
        if a(k)~=0
            P(k,a(k))=1;
        else
            P(k,:)=1/9;
        end;
    end;
end;

%constraint matrix (each row is a constraint)
C=[[1:9];
    [1:9]+9*1;
    [1:9]+9*2;
    [1:9]+9*3;
    [1:9]+9*4;
    [1:9]+9*5;
    [1:9]+9*6;
    [1:9]+9*7;
    [1:9]+9*8;
    1:9:81;
    2:9:81;
    3:9:81;
    4:9:81;
    5:9:81;
    6:9:81;
    7:9:81;
    8:9:81;
    9:9:81;
    [1:3 10:12 19:21]+3*0+27*0;
    [1:3 10:12 19:21]+3*1+27*0;
    [1:3 10:12 19:21]+3*2+27*0;
    [1:3 10:12 19:21]+3*0+27*1;
    [1:3 10:12 19:21]+3*1+27*1;
    [1:3 10:12 19:21]+3*2+27*1;
    [1:3 10:12 19:21]+3*0+27*2;
    [1:3 10:12 19:21]+3*1+27*2;
    [1:3 10:12 19:21]+3*2+27*2];

B=zeros(9);

imagesc(P');colormap(gray(256));axis('image');disp(['Press any key']);pause;

test=1;
for k=1:10000000;
    c=1+mod(k,27);

    for l=1:100;
        P(C(c,:),:)=P(C(c,:),:)./(ones(9,1)*sum(P(C(c,:),:)));
        P=P./(sum(P')'*ones(1,9));
    end;

    if (mod(k,400)==0)|0;

        [a,b]=max(P');
        B(:)=b;

        imagesc(P');colormap(gray(256));axis('image');drawnow;
        %P=get(get(gca,'children'),'Cdata');
        disp(['There are ' num2str(vss(B)) ' constraints not satisfied!']);

        if (vss(B)==0)&test;
            disp(['Found a solution in ' num2str(k) ' iterations']);
            disp(B);
            if ~(input('Continue y/n? ','s')=='y')
                break;
            end;
            test=0;
        end;
    end;
end;