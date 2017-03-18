function out = sigmoid_prime(z)
% "Derivative of the sigmoid function."

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% sigmoid prime
out = sigmoid(z).*(1-sigmoid(z));
end
