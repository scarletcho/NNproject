function [a, z] = feedforward(input, weights, biases, activationF)

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% feedforward based on different activation functions
switch activationF
    case 'softmax'
        % feedforward using softmax function
        z = weights'*input + repmat(biases,[1,size(input,2)]);    % weighted input
        a = softmax(z);                 % activation
        
    case 'sigmoid'
        % feedforward using sigmoid function
        z = weights'*input + repmat(biases,[1,size(input,2)]);    % weighted input
        a = sigmoid(z);                 % activation
end
end