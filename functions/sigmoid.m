function out = sigmoid(z)
% "Sigmoid activation function."

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% sigmoid
out = 1./(1+exp(-z));
end