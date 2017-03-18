function [param] = NaiveBayesClass(train_x,train_y)
% This function calculates parameters for Naive Bayesian classifier
% With the calculated 'param', you can test it out with your own test data
% Input
%   data: all x,y data set.
%         The row of x should be features, and
%         the column of x should be samples.
%         Corresponding labels should be appended as another column.
%
% Output
%   param: param is a structure variable which contains the following information
%       param.mu: mu matrix for all features by class
%       param.sigma: sigma matrix for all features by class
%       param.num_class: number of unique classes
%       param.unique_class: unique classes
%       param.num_feature: number of input features
%       param.class_prob: class probability
%%
unique_class = unique(train_y);
num_class = size(unique_class,2);
num_feature = size(train_x',2);

% P(X,Y) = P(Y) Pi P(Xk | Y)

% (1) Calculate Class probability
for i = 1:num_class
    class_prob(i) = ...
        sum(double(train_y==unique_class(i))) / length(train_y);
end
fprintf('Class probability is calculated\n')


% (2) Calculate Likelihood
%     Gaussian distribution parameters from training set
for i = 1:num_class
    xi = train_x(:,(train_y==unique_class(i)));
    mu(:,i) = mean(xi,2);
end

sigma = std(train_x,0,2);
fprintf('Likelihood parameters are calculated\n')

param = struct();

param.unique_class = unique_class;
param.num_class = num_class;
param.num_feature = num_feature;
param.mu = mu;
param.sigma = sigma;
param.class_prob = class_prob;
param.class_dist = size(train_y,2)*class_prob;
end