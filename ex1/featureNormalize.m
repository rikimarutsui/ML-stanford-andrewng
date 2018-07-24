function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2)); % Avg Value of the X
sigma = zeros(1, size(X, 2)); % The value of the X of (Max - Min)

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%   
    
for i=1:length(mu),
  mu(i) = mean(X_norm(:,i));         % Mean / Avg function = (sum(X_norm(:, i))) / length(X_norm(:, i));
  sigma(i) = std(X_norm(:,i));       % Standard Deviation = std (x) = sqrt ( 1/(N-1) SUM_i (x(i) - mean(x))^2 )
  X_norm(:, i) = (X_norm(:, i) - mu(i)) / sigma(i); 
  
  %for j=1:length(X_norm(:, i)),
   % X_norm(j, i) = (X_norm(j, i) - mu(i)) / sigma(i); 
  %endfor
endfor










% ============================================================

end
