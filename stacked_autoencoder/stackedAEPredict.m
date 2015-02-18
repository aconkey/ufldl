function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, x)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% x: Our matrix containing the training data as columns.  So, x(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

m = size(x,2);

a2 = sigmoid((stack{1}.w * x) + repmat(stack{1}.b,1,m));
a3 = sigmoid((stack{2}.w * a2) + repmat(stack{2}.b,1,m));

softmaxTheta_a3 = softmaxTheta * a3;
softmaxTheta_a3 = bsxfun(@minus, softmaxTheta_a3, max(softmaxTheta_a3, ...
                                                  [], 1));
a4 = exp(softmaxTheta_a3);
a4 = bsxfun(@rdivide, a4, sum(a4));

[maxvals,pred] = max(a4);

% -----------------------------------------------------------

end

% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
