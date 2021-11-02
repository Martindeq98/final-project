function [B,P]=keen(A,P);

% dimension fo the puzzle
n = size(A, 1);

% if only one argument is given (A)
if nargin == 1;
    % initialize P, get A
    P = zeros(n^2, n);
    a = A(:);
    for k = 1:n^2;
        % all given entries, we put a 1
        if a(k) ~= 0
            P(k, a(k)) = 1;
        % all unknown entries, we give a 1 / n
        else
            P(k,:) = 1/n;
        end;
    end;
end;

% constraint matrix (each row is a constraint)
C=[ % one of each 1 to 9 in each row
    [1:n] + n * 0;
    [1:n] + n * 1;
    [1:n] + n * 2;
    [1:n] + n * 3;
    [1:n] + n * 4;
    [1:n] + n * 5;
    [1:n] + n * 6;
    [1:n] + n * 7;
    [1:n] + n * 8;
    % one of each 1 to 9 in each column
    1:n:n^2;
    2:n:n^2;
    3:n:n^2;
    4:n:n^2;
    5:n:n^2;
    6:n:n^2;
    7:n:n^2;
    8:n:n^2;
    9:n:n^2;
    % constraints must be satisfied
    
    % sum constraints
%     [1:3 10:12 19:21] + 3 * 0 + 27 * 0;
    [1:3 10:12 19:21] + 3 * 1 + 27 * 0;
    [1:3 10:12 19:21] + 3 * 2 + 27 * 0;
    [1:3 10:12 19:21] + 3 * 0 + 27 * 1;
    [1:3 10:12 19:21] + 3 * 1 + 27 * 1;
    [1:3 10:12 19:21] + 3 * 2 + 27 * 1;];
%     [1:3 10:12 19:21] + 3 * 0 + 27 * 2;
%     [1:3 10:12 19:21] + 3 * 1 + 27 * 2;
%     [1:3 10:12 19:21] + 3 * 2 + 27 * 2];

% initialize B to empty
B = zeros(9);

% draw P
imagesc(P');colormap(gray(256));axis('image');disp(['Press any key']);pause;

% iterationg 10 million times over 18 constraints
for k = 1:10000000;
    % change constraint
    c = 1 + mod(k,23);

    % for 100 iterations
    for l = 1:100;
        % sinkhorn balance; normalize rows
        P(C(c ,:), :) = P(C(c, :), :) ./ (ones(9, 1) * sum(P(C(c ,:) ,:)));
        % normalize columns
        P = P ./ (sum(P')' * ones(1,9));
    end;

    % every 400 iterations of outerloop
    if (mod(k, 400) == 0) | 0;

        % get matrix A, B
        % we get the maximum entries in P
        [a, b] = max(P');
        B(:) = b;

        disp(max(P'))
        % draw results
        imagesc(P');colormap(gray(256));axis('image');drawnow;
        
        % check how many constraints are not satisfied
        disp(['There are ' num2str(vss(B)) ' constraints not satisfied!']);

        % if all constraints are satisfied
        if (vss(B)==0);
            % notify, show the solution, stop
            disp(['Found a solution in ' num2str(k) ' iterations']);
            disp(B);
            disp(P);
            break;
        end;
    end;
end;