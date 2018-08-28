function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_steps     = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_steps = C_steps;

error_i = [];
error_j = [];

for i = 1:size(C_steps, 1),
  C = C_steps(i);
  for j = 1:size(sigma_steps, 1),
    sigma = sigma_steps(j);
    
    % Train the SVM model
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    
    % Compute Prediction Error
    error = mean(double(predictions ~= yval));
    
    % Put error to the error j vector
    if j == 1,
      error_j = error;
    else
      error_j = [error_j; error];
    endif
  endfor
  % Put error j vector to the error i matrix
  if i == 1,
      error_i = error_j;
  else
      error_i = [error_i error_j];
  endif
endfor

% find the minimum value and index of the error_i
[row, col] = find(error_i == min(min(error_i)));

% return C and sigma

C = C_steps(col);
sigma = sigma_steps(row);




% =========================================================================

end
