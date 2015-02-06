function [cost, grad] = softmaxCost(theta, k, n, lambda, x, y)

% k - the number of classes 
% n - the size of the input vector
% lambda - weight decay parameter
% x - the n x m input matrix, where each column x(:, i) corresponds to
%        a single test set
% y - an m x 1 matrix containing the labels corresponding for the input data

% Unroll the parameters from theta
theta = reshape(theta, k, n);						
											
m = size(x, 2);									

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

thetax = theta * x;
thetax = bsxfun(@minus, thetax, max(thetax, [], 1)); % prevent overflow
htheta = exp(thetax);
htheta = bsxfun(@rdivide, htheta, sum(htheta));
indicator = full(sparse(y, 1:m, 1));

weight_decay = (lambda / 2) * sum(sum(theta.^2));

cost = (-(1./m) * sum(sum(indicator .* log(htheta)))) + weight_decay; 
thetagrad = (-(1./m) * (indicator - htheta) * x') + (lambda * theta);

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
