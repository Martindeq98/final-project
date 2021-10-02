function c = vss(B);
% VSS    Valid Sudoku Solution
%     Check if the matrix B is a valid sudoku matrix

% Mike Rabbat, University of Wisconsin-Madison
% 8 November 2006
% rabbat@cae.wisc.edu

% Check that A is 9-by-9 matrix
[n1, n2] = size(B);
if ((n1 ~= 9) | (n2 ~= 9))
   disp('B must be a 9-by-9 matrix');
end

% Initialize the probability matrix
P = zeros(81,9);
b = B(:);
for i=1:81
   if (b(i) > 0)
       P(i,b(i)) = 1;
   else
       P(i,:) = 1/9;
   end
end

%% Initialize lists of cells involved in each constraint
I = zeros(27,9);
C = reshape([1:81]',9,9);
% Row constraints
I(1,:) = C(1,:);
I(2,:) = C(2,:);
I(3,:) = C(3,:);
I(4,:) = C(4,:);
I(5,:) = C(5,:);
I(6,:) = C(6,:);
I(7,:) = C(7,:);
I(8,:) = C(8,:);
I(9,:) = C(9,:);
% Column constraints
I(10,:) = C(:,1)';
I(11,:) = C(:,2)';
I(12,:) = C(:,3)';
I(13,:) = C(:,4)';
I(14,:) = C(:,5)';
I(15,:) = C(:,6)';
I(16,:) = C(:,7)';
I(17,:) = C(:,8)';
I(18,:) = C(:,9)';
% 3x3 block constraints
I(19,:) = reshape(C(1:3,1:3),1,9);
I(20,:) = reshape(C(1:3,4:6),1,9);
I(21,:) = reshape(C(1:3,7:9),1,9);
I(22,:) = reshape(C(4:6,1:3),1,9);
I(23,:) = reshape(C(4:6,4:6),1,9);
I(24,:) = reshape(C(4:6,7:9),1,9);
I(25,:) = reshape(C(7:9,1:3),1,9);
I(26,:) = reshape(C(7:9,1:3),1,9);
I(27,:) = reshape(C(7:9,1:3),1,9);

% Check that each constraint yields a permutation matrix
c = 0;
for m=1:27
   M = P(I(m,:),:);
   if (~all(sum(M,1)==ones(1,9)))
       c = c+1;
   elseif (~all(sum(M,2)==ones(9,1)))
       c = c+1;
   end
end

if (nargout == 0)
   if (c == 0)
       disp('This is a valid Sudoku grid');
   else
       disp('This is not a valid Sudoku grid');
   end
end