function [pred] = softmaxPredict(softmaxModel, x)

% softmaxModel - model trained using softmaxTrain
% x - the N x M input matrix, where each column x(:,i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

thetax = theta * x;
thetax = bsxfun(@minus, thetax, max(thetax, [], 1)); % prevent overflow
htheta = exp(thetax);
htheta = bsxfun(@rdivide, htheta, sum(htheta));
[maxvals,pred] = max(htheta);

% ---------------------------------------------------------------------

end

