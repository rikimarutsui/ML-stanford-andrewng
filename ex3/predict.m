function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m, 1), X];   % Add 5000 1 to 5000 400 = 5000 401
z2 = a1 * Theta1';      % Theta1 * a1, 5000 401 times 25 401(Transpose) 401 25 = 5000 25(z2)
a2 = sigmoid(z2);       % sigmoid return 5000 25(a2)
a2 = [ones(size(z2,1), 1), a2];  % Add 5000 1 to 5000 25 = 5000 26
z3 = a2 * Theta2';      % Theta2 * a2, 5000 26 times 10 26(Transpose) 26 10 = 5000 10(z3)
a3 = sigmoid(z3);       % sigmoid output 5000 10

[maxVal, maxValRow] = max(a3, [], 2); % return the maxVal and maxValRow
disp(size(maxValRow));
% max Value Row is 1 to 10(10 = 0), the higher probabilty of the number will be sorted out
p = maxValRow;

% =========================================================================
end
