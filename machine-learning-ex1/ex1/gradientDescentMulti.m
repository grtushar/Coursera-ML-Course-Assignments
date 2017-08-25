function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

	h = X * theta;
            %display(h);

	lamda = zeros(size(X, 2), 1);
            %display(lamda);

	for i = 1:m
		lamda = lamda + (h(i) - y(i)) * X(i,:)';
	end
            %fprintf("---------after-------------");
            %display(lamda);

	lamda = lamda / m;
	theta = theta - alpha * lamda;
            %fprintf("--------theta---------");
            %display(theta);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
