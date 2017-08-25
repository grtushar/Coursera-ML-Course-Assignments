function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%[mxValues, hypothesis] = max(hypothesis');
%hypothesis = hypothesis';

%display(size(hypothesis));
%display(size(y));
%display(hypothesis(m,:));
%display(y(m,:));


a1 = [ones(m, 1) X]; % adding bias col

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 =  [ones(size(a2, 1), 1) a2]; % adding bias col
z3 = a2 * Theta2';
a3 = hypothesis = sigmoid(z3);

%hiddenLayer = sigmoid(X * Theta1'); % calculating first layer hypothesis
%hiddenLayer = [ones(size(hiddenLayer, 1), 1) hiddenLayer]; % adding bias col
%hypothesis = sigmoid(hiddenLayer * Theta2'); % calculating output layer (hypothesis)

modY = zeros(m, num_labels); % modY = modified y
for i = 1: m
	for j = 1: num_labels
		if j == y(i)
			modY(i, j) = 1;
		end
	end
end

%display(modY(m,:));
%display(y(m));

%for i = 1: m
	%for j = 1: num_labels
		%J += modY(i,j) * log(hypothesis(i,j)) + (1 - modY(i,j)) * log(1 - hypothesis(i,j));
	%end
%end

J = sum(sum((modY .* log(hypothesis)) + ((1 - modY) .* log(1 - hypothesis))));
J = J * (1/m);
J = -1 * J;


% calculating & adding regularization term
theta1WithoutFirstcol = Theta1(:,2:size(Theta1, 2));
theta2WithoutFirstcol = Theta2(:,2:size(Theta2, 2));

regTerm = 0;
regTerm += sum(sum(theta1WithoutFirstcol .^ 2));
regTerm += sum(sum(theta2WithoutFirstcol .^ 2));
regTerm *= (lambda / (2 * m));

J += regTerm;


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

del3 = hypothesis - modY; % step 3
del2 = (del3 * Theta2) .* (a2 .* (1 - a2)); % step 4
del2 = del2(:,2:size(del2,2)); % omiting first col

DEL2 = del3' * a2; % step 5
DEL1 = del2' * a1; %step 5

DEL2 = DEL2 * (1/m);
DEL1 = DEL1 * (1/m);

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

DEL2 = [DEL2(:,1) (DEL2(:,2:size(DEL2,2)) + ((lambda / m) * Theta2(:,2:size(Theta2,2))))]; % adding ignoring bias units
DEL1 = [DEL1(:,1) (DEL1(:,2:size(DEL1,2)) + ((lambda / m) * Theta1(:,2:size(Theta1,2))))]; % adding ignoring bias units

Theta2_grad = DEL2;
Theta1_grad = DEL1;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


%{
for i = 1: m
	for j = 1: size(a2, 1)
		Theta2_grad(j) += a2(j,2:size(a2,2)) * del3(i)';
	end
	
	for j = 1: size(a1, 1)
		Theta1_grad(i,j) += a1(j,2:size(a1,2)) * del2(i)';
	end
end
%}