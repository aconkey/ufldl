
function [cost,grad] = sparseAutoencoderCost(theta, num_in, num_hid, ...
                                               lambda, rho, beta, data)

% num_in:   the number of input units (probably 64) 
% num_hid:  the number of hidden units (probably 25) 
% lambda:   weight decay parameter
% rho:      the desired average activation for the hidden units
% beta:     weight of sparsity penalty term
% data:     Our 64x10000 training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:num_hid*num_in), num_hid, num_in);
W2 = reshape(theta(num_hid*num_in+1:2*num_hid*num_in), num_in, num_hid);
b1 = theta(2*num_hid*num_in+1:2*num_hid*num_in+num_hid);
b2 = theta(2*num_hid*num_in+num_hid+1:end);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

m = size(data,2); % number of training examples
                  %a1 = data; y = data; % autoencoder: inputs = targets

% make a forward pass in order to compute average activations:
a2 = sigmoid(W1 * data + repmat(b1,1,m));
a3 = sigmoid(W2 * a2 + repmat(b2,1,m));

% compute sparsity term:
rho_hat = (1./m) * sum(a2,2);
sparse_delta = -(rho ./ rho_hat) + ((1 - rho) ./ (1-rho_hat));

% backpropagation:
delta3 = -(data - a3) .* a3 .* (1 - a3); 
delta2 = ((W2' * delta3) + beta .* repmat(sparse_delta,1,m)) .* a2 .* (1 - a2);

deltaW1 = delta2 * data';
deltaW2 = delta3 * a2';
deltab1 = sum(delta2,2);
deltab2 = sum(delta3,2); 

% update grads:
W1grad = ((1./m) * deltaW1) + (lambda * W1);
W2grad = ((1./m) * deltaW2) + (lambda * W2);
b1grad = (1./m) * deltab1;
b2grad = (1./m) * deltab2;

JTerm = (1./m) * sum((1./2) * sum((a3 - data).^2)); 
regTerm = (lambda ./ 2) * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
KLTerm = beta .* sum((rho * log(rho./rho_hat)) + ((1 - rho) * log((1 - rho)./(1 - rho_hat))));
cost = JTerm + regTerm + KLTerm;
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

