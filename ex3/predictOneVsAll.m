function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
% A = 
%   8 1 6
%   3 5 7
%   4 9 2
%
%    [Y, U] = max(A, [], 1); %= max(A)  = get max value of the 1st demension 
% Y = % get column views max value [1st dimension]
%   8 9 7
% U = % get the max value of Y - index in column 
%   1 3 2
%     max(A, [], 2) = 
%  
%   [Y, U] = max(A, [], 2); % get max value of the 2nd demension
% Y =  % get row views max value [2nd dimension]
%   8
%   7
%   9
% U = % return the max value of Y - index of row
%   1
%   3
%   2

% Get Sigmoid
s = sigmoid(X * all_theta');
disp(size(s));
% In the Max Function max("Input Matrix", "Empty block", "The Dimension of Input Matrix")
[maxVal, maxValRowNo] = max(s, [], 2);

p = maxValRowNo;

% =========================================================================


end
