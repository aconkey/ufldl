%% Sparse autoencoder train.m

%  Instructions
%  ------------
% 
%  TODO: complete the code in sampleIMAGES.m, sparseAutoencoderCost.m, 
%        and computeNumericalGradient.m. You do not need to change
%        the code in this file. 
%

%% =====================================================================
%% STEP -1: Add path dependencies

addpath ../dependencies/
addpath ../dependencies/minFunc/
addpath ../dependencies/mnist/

%%======================================================================
%% STEP 0: Set the relevant parameters values

visibleSize = 8*8;      % number of input units 
hiddenSize = 25;        % number of hidden units 
sparsityParam = 0.01;   % desired average activation of the hidden units.
lambda = 0.0001;        % weight decay parameter       
beta = 3;               % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset

patches = sampleIMAGES;
display_network(patches(:,randi(size(patches,2),200,1)),8);

%  Obtain random parameters theta
theta = initializeParameters(hiddenSize, visibleSize);

%%======================================================================
%% STEP 2: Implement sparseAutoencoderCost
%

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, patches);

%%======================================================================
%% STEP 3: Gradient Checking
%
% After you have implemented computeNumericalGradient.m, run the following: 
%checkNumericalGradient();

% Now we can use it to check your cost function and derivative calculations
% for the sparse autoencoder.  
numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
                                                  hiddenSize, lambda, ...
                                                  sparsityParam, beta, ...
                                                  patches), theta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients and ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 

%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
options.Method = 'lbfgs'; 
options.maxIter = 400;
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                                   theta, options);

%%======================================================================
%% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 
