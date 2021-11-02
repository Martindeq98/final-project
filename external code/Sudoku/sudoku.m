function [B,P]=sudoku(A,P);

% if only one argument is given (A)
if nargin == 1;
    % initialize P, get A
    P = zeros(81,9);
    a = A(:);
    for k = 1:81;
        % all given entries, we put a 1
        if a(k) ~= 0
            P(k, a(k)) = 1;
        % all unknown entries, we give a 1 / n
        else
            P(k,:) = 1/9;
        end;
    end;
end;

% constraint matrix (each row is a constraint)
C=[ % one of each 1 to 9 in each row
    [1:9] + 9 * 0;
    [1:9] + 9 * 1;
    [1:9] + 9 * 2;
    [1:9] + 9 * 3;
    [1:9] + 9 * 4;
    [1:9] + 9 * 5;
    [1:9] + 9 * 6;
    [1:9] + 9 * 7;
    [1:9] + 9 * 8;
    % one of each 1 to 9 in each column
    1:9:81;
    2:9:81;
    3:9:81;
    4:9:81;
    5:9:81;
    6:9:81;
    7:9:81;
    8:9:81;
    9:9:81;
    % one of each 1 to 9 in each block
    [1:3 10:12 19:21] + 3 * 0 + 27 * 0;
    [1:3 10:12 19:21] + 3 * 1 + 27 * 0;
    [1:3 10:12 19:21] + 3 * 2 + 27 * 0;
    [1:3 10:12 19:21] + 3 * 0 + 27 * 1;
    [1:3 10:12 19:21] + 3 * 1 + 27 * 1;
    [1:3 10:12 19:21] + 3 * 2 + 27 * 1;
    [1:3 10:12 19:21] + 3 * 0 + 27 * 2;
    [1:3 10:12 19:21] + 3 * 1 + 27 * 2;
    [1:3 10:12 19:21] + 3 * 2 + 27 * 2];

% initialize B to empty
B = zeros(9);

% draw P
imagesc(P');colormap(gray(256));axis('image');disp(['Press any key']);pause;

% iterationg 10 million times over 27 constraints
for k = 1:10000000;
    % change constraint
    c = 1 + mod(k,27);

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