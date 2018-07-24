function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
x = X(:,2);        % Get the matrix X , 2nd Column
theta0 = theta(1); % Get vector theta, 1st value               
theta1 = theta(2); % Get vector theta, 2nd value
h = theta0 + (theta1 * x); % The hypothesis formula

theta0 = theta0 - alpha * (1/m) * sum(h - y); % Gradient Descent for theta0
theta1 = theta1 - alpha * (1/m) * sum((h - y) .* x); % Gradient Descent for theta1

theta = [theta0; theta1]; % simultaneously update

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta); % doing cost function
    
end

end
