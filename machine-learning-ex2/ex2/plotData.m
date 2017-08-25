function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


oneIdxs = find(y == 1);
zeroIdxs = find(y == 0);

plot(X(oneIdxs, 1), X(oneIdxs, 2), 'k+', 'MarkerSize', 10);
plot(X(zeroIdxs, 1), X(zeroIdxs, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 10);


% =========================================================================



hold off;

end
