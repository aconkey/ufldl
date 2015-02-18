function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: training data as columns. data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1 : numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

rho = 0.05;
beta = 3;

% forward pass: 
a2 = sigmoid(stack{1}.w * data + repmat(stack{1}.b,1,m));
a3 = sigmoid(stack{2}.w * a2 + repmat(stack{2}.b,1,m));

% sparsity terms:
rho_hat = (1./m) * sum(a2,2);
sparse_delta = -(rho ./ rho_hat) + ((1 - rho) ./ (1-rho_hat));

% softmax:
softmaxTheta_a3 = softmaxTheta * a3;
softmaxTheta_a3 = bsxfun(@minus, softmaxTheta_a3, max(softmaxTheta_a3, ...
                                                  [], 1));
a4 = exp(softmaxTheta_a3);
a4 = bsxfun(@rdivide, a4, sum(a4));

% softmaxTheta:
softmaxThetaGrad = (-(1./m) * (groundTruth - a4) * a3') + (lambda * ...
                                                  softmaxTheta);

% backprop: 
delta3 = -(softmaxTheta' * (groundTruth - a4)) .* a3 .* (1 - a3);
delta2 = ((stack{2}.w' * delta3) + beta .* repmat(sparse_delta,1,m)) .* a2 .* (1 -a2);

deltaW1 = delta2 * data';
deltaW2 = delta3 * a2';
deltab1 = sum(delta2,2);
deltab2 = sum(delta3,2);

stackgrad{1}.w = ((1./m) * deltaW1) + (lambda * stack{1}.w);
stackgrad{1}.b = (1./m) * deltab1;
stackgrad{2}.w = ((1./m) * deltaW2) + (lambda * stack{2}.w);
stackgrad{2}.b = (1./m) * deltab2;

JTerm = (-(1./m) * sum(sum(groundTruth .* log(a4))));
weight_decay = (lambda / 2) * sum(sum(softmaxTheta .^ 2));
regTerm = (lambda ./ 2) * (sum(sum(stack{1}.w .^2)) + sum(sum(stack{2}.w ...
                                                  .^2)));
KLTerm = beta .* sum((rho * log(rho./rho_hat)) + ((1 - rho) * log((1 - rho)./(1 - rho_hat))));
cost = JTerm + weight_decay + regTerm + KLTerm;
% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end