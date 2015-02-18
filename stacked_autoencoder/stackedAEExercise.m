%% Stacked Autoencoder

%  Instructions
%  ------------
% 
%  Complete code in stackedAECost.m
%

%% =====================================================================
%% STEP -1: Add path dependencies                                       
addpath ../dependencies/
addpath ../dependencies/minFunc/
addpath ../dependencies/mnist/
addpath ../sparse_autoencoder/
addpath ../softmax/
addpath ../stl_exercise/

%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will     
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database                            
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');

% Remap 0 to 10 since our labels need to start from 1:
trainLabels(trainLabels == 0) = 10; 

DEBUG = false;
if DEBUG
   inputSize = 8;
   hiddenSizeL1 = 10;
   hiddenSizeL2 = 10;
   trainData = randn(8,100);
   trainLabels = randi(10,100,1);
end

%%======================================================================
%% STEP 2: Train the first sparse autoencoder                           
%  Train the first sparse autoencoder on the unlabelled STL training images.

% Randomly initialize the parameters:
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                a hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta

options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';

[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                                   sae1Theta, options);

% LOAD data instead of running code above; memory issues: 
%load sae1OptTheta.mat

%%======================================================================
%% STEP 2: Train the second sparse autoencoder                          
%  Train the second sparse autoencoder on the first autoencoder features.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';

[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                                   sae2Theta, options);


% LOAD DATA instead of running code above; memory can't handle it:
%load sae2OptTheta.mat

%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);
% -------------------------------------------------------------------------

softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options)

saeSoftmaxOptTheta = softmaxModel.optTheta(:);

%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%%===========================================================================
%% Gradient Checking                                                         



if DEBUG
   [cost, grad] = stackedAECost(stackedAETheta, inputSize, hiddenSizeL2, ...
                                               numClasses, netconfig, ...
                                               lambda, trainData, trainLabels); 
    
   numGrad = computeNumericalGradient(@(x) stackedAECost(x, inputSize, ...
                                                     hiddenSizeL2, ...
                                                     numClasses, ...
                                                     netconfig, ...
                                                     lambda, trainData, ...
                                                     trainLabels), ...
                                                     stackedAETheta);
   disp([numGrad, grad]);
   diff = norm(numGrad - grad) / norm(numGrad + grad);
   disp(diff);
end

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%

options.Method = 'lbfgs'; 
options.maxIter = 400;
options.display = 'on';

[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                   inputSize, hiddenSizeL2, numClasses, ...
                                   netconfig, lambda, trainData, ...
                                   trainLabels), stackedAETheta, options);
 
%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
